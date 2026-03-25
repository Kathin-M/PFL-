{-# LANGUAGE RecordWildCards #-}

-- | PFL – Main entry-point.
--   Loads CICIDS2017 benign samples, trains the SimCLR encoder,
--   and saves the learned weights for later anomaly detection.
module Main where

import System.Directory         (doesFileExist)
import Torch

import DataPipeline             (loadBenignSamples)
import qualified DataPipeline   as DP
import SimCLR                   (SimCLRModel(..), initSimCLR, encodeForward)
import Training                 (trainLoop, defaultConfig, TrainConfig(..))
import Evaluation               (evaluateModel)

-- | Path to the local CICIDS2017 CSV.
defaultCSVPath :: FilePath
defaultCSVPath = "cicids2017_cleaned.csv"

-- | Where to save the trained model weights.
weightsPath :: FilePath
weightsPath = "pfl_encoder_weights.pt"

main :: IO ()
main = do
  putStrLn "============================================="
  putStrLn "  PFL - SimCLR Unsupervised NIDS Training"
  putStrLn "============================================="
  putStrLn ""

  -- 1. Check the dataset exists
  exists <- doesFileExist defaultCSVPath
  if not exists
    then error $ "Dataset not found at: " ++ defaultCSVPath
              ++ "\nMake sure Google Drive is mounted and the file is at that path."
    else putStrLn $ "Dataset found: " ++ defaultCSVPath

  -- 2. Load benign samples
  putStrLn "\n[1/3] Loading and filtering benign samples..."
  samples <- loadBenignSamples defaultCSVPath

  -- 3. Initialise model
  putStrLn "\n[2/3] Initialising SimCLR model..."
  model0 <- initSimCLR DP.numFeatures 32 16
  putStrLn $  "  Encoder:    " ++ show DP.numFeatures ++ " -> 128 -> 64 -> 32"
  putStrLn    "  Projection: 32 -> 16"
  putStrLn    "  Optimiser:  GD (Gradient Descent)"

  -- 4. Train
  putStrLn "\n[3/3] Training..."
  let config = defaultConfig
        { cfgEpochs    = 20
        , cfgBatchSize = 256
        }
  trainedModel <- trainLoop config samples model0

  -- 5. Save encoder weights using Hasktorch's saveParams
  putStrLn "\nSaving model weights..."
  saveParams trainedModel weightsPath
  putStrLn $ "  Model saved to: " ++ weightsPath

  -- 6. Done
  putStrLn ""
  putStrLn "============================================="
  putStrLn "  Training complete!"
  putStrLn ""
  putStrLn "  To use the encoder for anomaly detection:"
  putStrLn "    1. Load weights with loadParams"
  putStrLn "    2. Feed new traffic through encodeForward"
  putStrLn "  4. Samples far from centroid = anomalies"
  putStrLn "============================================="

  -- 7. Evaluate Model
  -- We set a normalized distance threshold of 0.91 (between Benign 0.83 and Anomaly 0.99)
  evaluateModel defaultCSVPath trainedModel 0.91
