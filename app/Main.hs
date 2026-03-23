{-# LANGUAGE ScopedTypeVariables #-}

-- | PFL – Main entry-point.
--   Loads CICIDS2017 benign samples, trains the SimCLR encoder,
--   and saves the learned weights for later anomaly detection.
module Main where

import System.Directory         (doesFileExist)
import Torch

import DataPipeline             (loadBenignSamples, numFeatures)
import SimCLR                   (SimCLRModelSpec(..), SimCLRModel(..)
                                , encodeForward)
import Training                 (trainLoop, defaultConfig, TrainConfig(..))

-- | Path to the CICIDS2017 CSV on Google Drive (mounted in Colab).
defaultCSVPath :: FilePath
defaultCSVPath = "/content/drive/MyDrive/PFL/cicids2017_cleaned.csv"

-- | Where to save the trained encoder weights.
weightsPath :: FilePath
weightsPath = "pfl_encoder_weights.pt"

main :: IO ()
main = do
  putStrLn "============================================="
  putStrLn "  PFL – SimCLR Unsupervised NIDS Training"
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
  let spec = SimCLRModelSpec
        { smInputDim  = numFeatures   -- 52
        , smLatentDim = 32
        , smProjDim   = 16
        }
  model0 <- sample spec
  putStrLn $  "  Encoder:    " ++ show numFeatures ++ " → 128 → 64 → 32"
  putStrLn    "  Projection: 32 → 16"
  putStrLn    "  Optimiser:  SGD"

  -- 4. Train
  putStrLn "\n[3/3] Training..."
  let config = defaultConfig
        { cfgEpochs    = 20
        , cfgBatchSize = 256
        , cfgLR        = 1e-3
        }
  trainedModel <- trainLoop config samples model0

  -- 5. Save encoder weights
  putStrLn "\nSaving encoder weights..."
  let encoderParams = flattenParameters (smEncoder trainedModel)
  save encoderParams weightsPath
  putStrLn $ "✓ Encoder weights saved to: " ++ weightsPath

  -- 6. Done
  putStrLn ""
  putStrLn "============================================="
  putStrLn "  Training complete!"
  putStrLn ""
  putStrLn "  To use the encoder for anomaly detection:"
  putStrLn "    1. Load the weights with  Torch.load"
  putStrLn "    2. Feed new traffic through encodeForward"
  putStrLn "    3. Compute the reconstruction / distance score"
  putStrLn "    4. Samples far from the benign cluster = anomalies"
  putStrLn "============================================="
