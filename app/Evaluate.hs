{-# LANGUAGE RecordWildCards #-}

-- | PFL – Evaluation entry-point.
--   Loads the pre-trained SimCLR encoder weights and dynamically
--   tests the efficiency on the dataset without retraining.
module Main where

import System.Directory         (doesFileExist)
import Torch

import qualified DataPipeline   as DP
import SimCLR                   (SimCLRModel(..), initSimCLR)
import Evaluation               (evaluateModel)

-- | Path to the local CICIDS2017 CSV.
defaultCSVPath :: FilePath
defaultCSVPath = "cicids2017_cleaned.csv"

-- | Where to load the trained model weights from.
weightsPath :: FilePath
weightsPath = "pfl_encoder_weights.pt"

main :: IO ()
main = do
  putStrLn "============================================="
  putStrLn "  PFL - SimCLR Unsupervised NIDS Evaluation"
  putStrLn "============================================="
  putStrLn ""

  csvExists <- doesFileExist defaultCSVPath
  if not csvExists
    then error $ "Dataset not found at: " ++ defaultCSVPath
    else putStrLn $ "Dataset found: " ++ defaultCSVPath

  weightsExist <- doesFileExist weightsPath
  if not weightsExist
    then error $ "Trained weights not found at: " ++ weightsPath ++ "\nPlease run 'cabal run pfl-train' (or 'runghc -isrc app/Main.hs') first!"
    else putStrLn $ "Weights found: " ++ weightsPath

  putStrLn "\nLoading SimCLR model architecture..."
  model0 <- initSimCLR DP.numFeatures 32 16
  
  putStrLn "Loading trained weights..."
  trainedModel <- loadParams model0 weightsPath

  -- Run Evaluation with a normalized distance threshold of 0.91 (between Benign 0.83 and Anomaly 0.99)
  evaluateModel defaultCSVPath trainedModel 0.91
