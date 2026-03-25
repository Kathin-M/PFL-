{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE BangPatterns #-}

-- | Evaluates the trained SimCLR model's anomaly detection efficiency.
module Evaluation
  ( evaluateModel
  ) where

import           Torch
import           SimCLR          (SimCLRModel, encodeForward, smEncoder)
import           DataPipeline    (parseAnyRow, batchTensor, numFeatures)

import qualified Data.ByteString.Lazy  as BL
import qualified Data.Csv.Streaming    as CsvS
import qualified Data.Csv              as Csv
import qualified Data.Vector           as V
import           Data.List             (foldl')
import           Text.Printf           (printf)
import           Control.Monad         (when)
import           System.IO             (hFlush, stdout)

data ConfusionMatrix = CM
  { tp :: !Int   -- True Positive (Anomaly correctly identified)
  , fp :: !Int   -- False Positive (Benign identified as Anomaly)
  , tn :: !Int   -- True Negative (Benign correctly identified)
  , fn :: !Int   -- False Negative (Anomaly identified as Benign)
  , sumBenignDist  :: !Float -- Total distance of Benign samples
  , sumAnomalyDist :: !Float -- Total distance of Anomaly samples
  } deriving Show

emptyCM :: ConfusionMatrix
emptyCM = CM 0 0 0 0 0.0 0.0

-- | L2 normalize each row of a 2D tensor
l2Normalize :: Tensor -> Tensor
l2Normalize t =
  let norms = Torch.sqrt (sumDim (Dim 1) KeepDim Float (t * t) + asTensor (1e-8 :: Float))
  in t / norms

-- | Calculate the benign centroid using a small batch of known benign data
computeCentroid :: SimCLRModel -> [[Float]] -> Tensor
computeCentroid model benignSamples =
  let t      = batchTensor benignSamples
      z      = encodeForward (smEncoder model) t
      zn     = l2Normalize z
      -- zn is [B, LatentDim], average along batch dimension (Dim 0)
      rawCentroid = meanDim (Dim 0) KeepDim Float zn
      -- The centroid itself must be l2 normalized so it lives on the hypersphere
      centroid = l2Normalize rawCentroid
  in centroid

-- | Evaluate distance and return the predicted anomaly status
isAnomalyPrediction :: Tensor -> Tensor -> Float -> Bool
isAnomalyPrediction centroid z threshold =
  let diff = z - centroid
      dist = Torch.sqrt (sumAll (diff * diff))  -- Euclidean
      distVal = asValue (Torch.toDType Float dist) :: Float
  in distVal > threshold

-- | Run streaming evaluation over the dataset
evaluateModel :: FilePath -> SimCLRModel -> Float -> IO ()
evaluateModel fp model threshold = do
  putStrLn "\n============================================="
  putStrLn "  Evaluation Mode (Streaming)"
  putStrLn "============================================="
  putStrLn $ "  Threshold: " ++ show threshold

  raw <- BL.readFile fp
  case CsvS.decodeByName raw of
    Left err -> error $ "Evaluation parse error: " ++ err
    Right (_hdr, stream) -> do
      
      -- First, we need to extract a small batch of benign samples to compute the centroid
      let (benignBatch, restStream) = extractBenignBatch 1000 stream []
      
      putStrLn $ "  Computing Benign Centroid using " ++ show (length benignBatch) ++ " samples..."
      let centroid = computeCentroid model benignBatch
      
      putStrLn   "  Streaming dataset for evaluation... (This will print progress every 100k samples)"
      finalCM <- evaluateStream restStream model centroid threshold emptyCM
      
      printMetrics finalCM

-- | Manually extract a few benign samples from the front of the stream
extractBenignBatch :: Int -> CsvS.Records Csv.NamedRecord -> [[Float]] -> ([[Float]], CsvS.Records Csv.NamedRecord)
extractBenignBatch 0 stream acc = (acc, stream)
extractBenignBatch n (CsvS.Cons (Right nr) rs) acc
  | Just (fs, False) <- parseAnyRow nr = extractBenignBatch (n - 1) rs (fs : acc)
  | otherwise                          = extractBenignBatch n rs acc
extractBenignBatch n (CsvS.Cons (Left _) rs) acc = extractBenignBatch n rs acc
extractBenignBatch _ stream@(CsvS.Nil _ _) acc = (acc, stream)

-- | Process a single batch of anomalies mathematically
processBatch :: [([Float], Bool)] -> SimCLRModel -> Tensor -> Float -> ConfusionMatrix -> IO ConfusionMatrix
processBatch buf model centroid threshold !cm = do
  let (features, actualIsAnomaly) = unzip buf
      t = batchTensor features
      z = encodeForward (smEncoder model) t
      zn = l2Normalize z
      
      -- Calculate distance: diff is [B, LatentDim]
      diff = zn - centroid
      
      -- sumDim (Dim 1) -> shape [B]
      distsSq = sumDim (Dim 1) RemoveDim Float (diff * diff)
      dists = Torch.sqrt distsSq
      
      preds :: [Float]
      preds = asValue (Torch.toDType Float dists)

      -- Thresholding
      predIsAnomaly = map (> threshold) preds
  
  -- Fold over everything and tally confusion matrix
  let nextCM = foldl' (\curr (actual, (pred, dist)) -> updateCM actual pred dist curr) cm (zip actualIsAnomaly (zip predIsAnomaly preds))
  return nextCM

-- | Run the rest of the stream, batching 2048 at a time for efficiency, and updating the Confusion Matrix
evaluateStream :: CsvS.Records Csv.NamedRecord -> SimCLRModel -> Tensor -> Float -> ConfusionMatrix -> IO ConfusionMatrix
evaluateStream stream model centroid threshold cm = go stream cm (0 :: Int) []
  where
    -- Short-circuit when we hit half a million samples
    go _ !currCM !count !buf | count >= 500000 =
      if null buf then return currCM else processBatch (reverse buf) model centroid threshold currCM
      
    go (CsvS.Cons (Right nr) rs) (!currCM) !count !buf =
      case parseAnyRow nr of
        Just (fs, actualIsAnomaly) -> do
          let buf' = (fs, actualIsAnomaly) : buf
              newCount = count + 1
          
          if length buf' >= 2048
            then do
              nextCM <- processBatch (reverse buf') model centroid threshold currCM
              putStr $ "\r    Processed " ++ show newCount ++ " samples...  [Batching in progress]"
              hFlush stdout
              go rs nextCM newCount []
            else go rs currCM newCount buf'
        Nothing -> go rs currCM count buf
    
    go (CsvS.Cons (Left _) rs) !currCM !count !buf = go rs currCM count buf
    go (CsvS.Nil _ _) !currCM count buf
      | null buf  = return currCM
      | otherwise = processBatch (reverse buf) model centroid threshold currCM

updateCM :: Bool -> Bool -> Float -> ConfusionMatrix -> ConfusionMatrix
updateCM True  predIsAnom dist cm = 
  let cm1 = cm { sumAnomalyDist = sumAnomalyDist cm + dist }
  in if predIsAnom then cm1 { tp = tp cm1 + 1 } else cm1 { fn = fn cm1 + 1 }

updateCM False predIsAnom dist cm = 
  let cm1 = cm { sumBenignDist = sumBenignDist cm + dist }
  in if predIsAnom then cm1 { fp = fp cm1 + 1 } else cm1 { tn = tn cm1 + 1 }

printMetrics :: ConfusionMatrix -> IO ()
printMetrics (CM tpVal fpVal tnVal fnVal sumBenDist sumAnomDist) = do
  let total = tpVal + fpVal + tnVal + fnVal
      accuracy = fromIntegral (tpVal + tnVal) / fromIntegral total :: Float
      
      precision = if (tpVal + fpVal) == 0 then 0.0 else fromIntegral tpVal / fromIntegral (tpVal + fpVal) :: Float
      recall    = if (tpVal + fnVal) == 0 then 0.0 else fromIntegral tpVal / fromIntegral (tpVal + fnVal) :: Float
      f1Score   = if (precision + recall) == 0 then 0.0 else 2 * (precision * recall) / (precision + recall) :: Float

  putStrLn "\n--- Results ---"
  putStrLn $ "  Total Tested:   " ++ show total
  putStrLn $ "  True Positives: " ++ show tpVal ++ " (Anomalies caught)"
  putStrLn $ "  False Positives:" ++ show fpVal ++ " (False alarms)"
  putStrLn $ "  True Negatives: " ++ show tnVal ++ " (Benign ignored)"
  putStrLn $ "  False Negatives:" ++ show fnVal ++ " (Anomalies missed)"
  
  let avgBenign  = if (tnVal + fpVal) == 0 then 0 else sumBenDist / fromIntegral (tnVal + fpVal)
      avgAnomaly = if (tpVal + fnVal) == 0 then 0 else sumAnomDist / fromIntegral (tpVal + fnVal)
      
  putStrLn ""
  printf "  Average Distance (Benign):   %.4f\n" avgBenign
  printf "  Average Distance (Anomaly):  %.4f\n" avgAnomaly
  putStrLn ""
  
  printf "  Accuracy:  %.2f%%\n" (accuracy * 100)
  printf "  Precision: %.2f%%\n" (precision * 100)
  printf "  Recall:    %.2f%%\n" (recall * 100)
  printf "  F1-Score:  %.2f%%\n" (f1Score * 100)
  putStrLn "============================================="
