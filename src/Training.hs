{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}

-- | Training loop for SimCLR:
--     * NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss
--     * Mini-batch contrastive training using Hasktorch's runStep + GD
module Training
  ( ntXentLoss
  , trainLoop
  , TrainConfig(..)
  , defaultConfig
  ) where

import qualified Data.Vector        as V
import           System.Random      (randomRIO)
import           Text.Printf        (printf)

import           Torch

import           DataPipeline       (augmentPair, batchTensor)
import qualified DataPipeline       as DP
import           SimCLR             (SimCLRModel(..), simclrForward)

-- ================================================================
--  Configuration
-- ================================================================

data TrainConfig = TrainConfig
  { cfgEpochs      :: Int
  , cfgBatchSize   :: Int
  , cfgLR          :: Tensor      -- ^ learning rate as a Tensor scalar
  , cfgTemperature :: Float       -- ^ tau for NT-Xent
  , cfgInputDim    :: Int
  , cfgLatentDim   :: Int
  , cfgProjDim     :: Int
  } deriving (Show)

defaultConfig :: TrainConfig
defaultConfig = TrainConfig
  { cfgEpochs      = 20
  , cfgBatchSize   = 256
  , cfgLR          = asTensor (1e-3 :: Float)
  , cfgTemperature = 0.5
  , cfgInputDim    = DP.numFeatures   -- 52
  , cfgLatentDim   = 32
  , cfgProjDim     = 16
  }

-- ================================================================
--  NT-Xent Loss (simplified but correct)
-- ================================================================

-- | Compute the NT-Xent contrastive loss.
ntXentLoss :: Float -> Tensor -> Tensor -> Tensor
ntXentLoss temperature z1 z2 =
  let
    -- L2-normalise along feature dim
    z1n = l2Normalize z1
    z2n = l2Normalize z2

    -- Cosine similarity between all pairs of z1 and z2: [B, B]
    simMatrix = matmul z1n (Torch.transpose (Dim 0) (Dim 1) z2n)
                  / asTensor temperature

    -- The positive pairs are exactly on the diagonal
    logProbs = logSoftmax (Dim 1) simMatrix
    
    batchSize = head (shape z1)
    
    -- Identity masking to extract the diagonal safely without using typeclass-heavy bounds
    identityMask = asTensor [ [ if i == j then (1.0 :: Float) else 0.0 | j <- [1..batchSize] ] 
                            | i <- [1..batchSize] ]
    
    -- Element-wise multiply clears off-diagonals, then sum over all elements
    sumPositives = sumAll (logProbs * identityMask)
    
    -- Cross-entropy is the negative mean of the log probabilities of the true classes (the diagonal)
    loss = -(sumPositives) / asTensor (fromIntegral batchSize :: Float)
  in loss

-- | L2 normalize each row of a 2D tensor
l2Normalize :: Tensor -> Tensor
l2Normalize t =
  let norms = Torch.sqrt (sumDim (Dim 1) KeepDim Float (t * t) + asTensor (1e-8 :: Float))
  in t / norms

-- ================================================================
--  Sampling a mini-batch
-- ================================================================

-- | Randomly sample batchSize indices from the data.
sampleBatch :: Int -> Int -> IO [Int]
sampleBatch dataSize batchSize =
  sequence [ randomRIO (0, dataSize - 1) | _ <- [1..batchSize] ]

-- ================================================================
--  Full training loop
-- ================================================================

-- | Train for the configured number of epochs, printing loss
--   after each epoch. Uses Hasktorch's runStep with GD optimizer.
trainLoop
  :: TrainConfig
  -> V.Vector [Float]
  -> SimCLRModel
  -> IO SimCLRModel
trainLoop cfg@TrainConfig{..} samples model0 = do
  putStrLn "Starting SimCLR training..."
  putStrLn $ "  Epochs:      " ++ show cfgEpochs
  putStrLn $ "  Batch size:  " ++ show cfgBatchSize
  putStrLn $ "  Temperature: " ++ show cfgTemperature
  putStrLn ""

  let n          = V.length samples
      numBatches = Prelude.max 1 (n `Prelude.div` cfgBatchSize)

  -- Training loop using foldLoop from Hasktorch
  (finalModel, _) <- foldLoop (model0, 0 :: Int) cfgEpochs $ \(model, _) epoch -> do
    -- Run one epoch: iterate over mini-batches
    (model', totalLoss) <- foldLoop (model, 0.0 :: Float) numBatches $ \(m, cumLoss) _ -> do
      -- Sample a mini-batch
      idxs <- sampleBatch n cfgBatchSize
      let rawBatch = map (samples V.!) idxs

      -- Augment each sample into two views
      pairs <- mapM augmentPair rawBatch
      let (view1s, view2s) = unzip pairs
          t1 = batchTensor view1s    -- [B, 52]
          t2 = batchTensor view2s    -- [B, 52]

      -- Forward + loss
      let z1   = simclrForward m t1
          z2   = simclrForward m t2
          loss = ntXentLoss cfgTemperature z1 z2

      -- SGD step using Hasktorch's runStep
      (m', _) <- runStep m GD loss cfgLR
      let lossVal = asValue (Torch.toDType Float loss) :: Float
      return (m', cumLoss + lossVal)

    let avgLoss = totalLoss / fromIntegral numBatches
    printf "  Epoch %3d / %d  |  Loss: %.6f\n" epoch cfgEpochs avgLoss
    return (model', epoch)

  return finalModel
