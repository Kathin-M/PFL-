{-# LANGUAGE ScopedTypeVariables #-}

-- | Training loop for SimCLR:
--     • NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss
--     • Mini-batch contrastive training with SGD
module Training
  ( ntXentLoss
  , trainEpoch
  , trainLoop
  , TrainConfig(..)
  , defaultConfig
  ) where

import           Data.List          (foldl')
import qualified Data.Vector        as V
import           System.Random      (randomRIO)
import           Text.Printf        (printf)

import           Torch

import           DataPipeline       (augmentPair, batchTensor, numFeatures)
import           SimCLR             (SimCLRModel(..), SimCLRModelSpec(..)
                                    , simclrForward, encodeForward)

-- ================================================================
--  Configuration
-- ================================================================

data TrainConfig = TrainConfig
  { cfgEpochs      :: !Int
  , cfgBatchSize   :: !Int
  , cfgLR          :: !Float       -- ^ learning rate
  , cfgTemperature :: !Float       -- ^ τ for NT-Xent
  , cfgInputDim    :: !Int
  , cfgLatentDim   :: !Int
  , cfgProjDim     :: !Int
  } deriving (Show)

defaultConfig :: TrainConfig
defaultConfig = TrainConfig
  { cfgEpochs      = 20
  , cfgBatchSize   = 256
  , cfgLR          = 1e-3
  , cfgTemperature = 0.5
  , cfgInputDim    = numFeatures   -- 52
  , cfgLatentDim   = 32
  , cfgProjDim     = 16
  }

-- ================================================================
--  NT-Xent Loss
-- ================================================================

-- | Compute the NT-Xent contrastive loss.
--
--   @z1@ and @z2@ are [B, D] tensors  (two augmented views).
--
--   For each sample i, the positive pair is (z1_i, z2_i).
--   All other 2(B-1) samples in the combined batch are negatives.
ntXentLoss :: Float -> Tensor -> Tensor -> Tensor
ntXentLoss temperature z1 z2 =
  let
    -- L2-normalise along feature dim
    z1n = normalize z1
    z2n = normalize z2
    -- Combined representations: [2B, D]
    z   = cat (Dim 0) [z1n, z2n]              -- [2B, D]
    n   = head (shape z1n)                     -- B
    -- Similarity matrix: [2B, 2B]
    sim = matmul z (transpose2D z) / asTensor temperature
    -- Mask out self-similarity (diagonal)
    mask  = Torch.toDType Bool
          $ (1 :: Float) `Torch.sub` eyeSquare (2 * n)          -- [2B, 2B]
    sim'  = maskedFill sim mask (asTensor (-1e9 :: Float))
    -- For the first B rows, the positive is at column (i + B).
    -- For rows B..2B-1, the positive is at column (i - B).
    posTop    = diag (slice 1 n (2*n) 1 sim) 0           -- [B] upper-right block diag
    posBottom = diag (slice 1 0 n     1 (Torch.indexSelect 0 (asTensor [(fromIntegral n :: Int) .. (fromIntegral (2*n - 1) :: Int)]) sim)) 0
    positives = cat (Dim 0) [posTop, posBottom]           -- [2B]
    -- log-sum-exp denominator per row
    expSim = Torch.exp sim'
    denom  = Torch.log (sumDim (Dim 1) RemoveDim Float expSim)   -- [2B]
    -- loss  =  -mean( positive / τ  –  log Σ exp(sim/τ) )
    loss   = negate' (Torch.mean (positives / asTensor temperature - denom))
  in loss
  where
    normalize t =
      let n = Torch.sqrt (sumDim (Dim 1) KeepDim Float (t * t) + asTensor (1e-8 :: Float))
      in t / n
    negate' t = mulScalar (-1.0 :: Float) t
    transpose2D t = Torch.transpose (Dim 0) (Dim 1) t

-- ================================================================
--  Sampling a mini-batch
-- ================================================================

-- | Randomly sample @batchSize@ indices from the data.
sampleIndices :: Int -> Int -> IO [Int]
sampleIndices dataSize batchSize =
  sequence [ randomRIO (0, dataSize - 1) | _ <- [1..batchSize] ]

-- ================================================================
--  One epoch
-- ================================================================

-- | Run one epoch:  iterate over the dataset in mini-batches,
--   compute NT-Xent, and do an SGD step.
trainEpoch
  :: TrainConfig
  -> V.Vector [Float]
  -> SimCLRModel
  -> IO (SimCLRModel, Float)       -- ^ updated model, avg epoch loss
trainEpoch cfg samples model = do
  let n          = V.length samples
      bs         = cfgBatchSize cfg
      numBatches = max 1 (n `div` bs)
  (model', totalLoss) <- foldlStep numBatches (model, 0.0)
  return (model', totalLoss / fromIntegral numBatches)
  where
    foldlStep 0 acc = return acc
    foldlStep remaining (m, cumLoss) = do
      idxs <- sampleIndices (V.length samples) (cfgBatchSize cfg)
      let rawBatch = map (samples V.!) idxs         -- [[Float]]
      -- Augment each sample into two views
      pairs <- mapM augmentPair rawBatch
      let (view1s, view2s) = unzip pairs
          t1 = batchTensor view1s                    -- [B, 52]
          t2 = batchTensor view2s                    -- [B, 52]
      -- Forward
      let z1  = simclrForward m t1
          z2  = simclrForward m t2
          loss = ntXentLoss (cfgTemperature cfg) z1 z2
      -- Backprop
      let params   = flattenParameters m
          grads    = grad loss params
          params'  = zipWith (\p g -> p - mulScalar (cfgLR cfg) g) params grads
          m'       = replaceParameters m params'
          lossVal  = asValue (Torch.toDouble loss) :: Double
      foldlStep (remaining - 1) (m', cumLoss + realToFrac lossVal)

-- ================================================================
--  Full training loop
-- ================================================================

-- | Train for the configured number of epochs, printing loss
--   after each epoch.
trainLoop
  :: TrainConfig
  -> V.Vector [Float]
  -> SimCLRModel
  -> IO SimCLRModel
trainLoop cfg samples model0 = do
  putStrLn "Starting SimCLR training..."
  putStrLn $ "  Epochs:      " ++ show (cfgEpochs cfg)
  putStrLn $ "  Batch size:  " ++ show (cfgBatchSize cfg)
  putStrLn $ "  LR:          " ++ show (cfgLR cfg)
  putStrLn $ "  Temperature: " ++ show (cfgTemperature cfg)
  putStrLn ""
  go 1 model0
  where
    go epoch model
      | epoch > cfgEpochs cfg = return model
      | otherwise = do
          (model', avgLoss) <- trainEpoch cfg samples model
          printf "  Epoch %3d / %d  |  Loss: %.6f\n" epoch (cfgEpochs cfg) avgLoss
          go (epoch + 1) model'
