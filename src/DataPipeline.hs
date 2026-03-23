{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE DeriveGeneric #-}

-- | Data pipeline for CICIDS2017.
--   Lazily parses the CSV, filters for "Normal Traffic", and
--   provides augmentation helpers for SimCLR.
module DataPipeline
  ( loadBenignSamples
  , augmentPair
  , batchTensor
  , numFeatures
  ) where

import           GHC.Generics          (Generic)
import qualified Data.ByteString.Lazy  as BL
import qualified Data.Csv              as Csv
import qualified Data.Vector           as V
import           Data.Text             (Text)
import qualified Data.Text             as T
import qualified Data.Text.Read        as TR
import           Data.Maybe            (mapMaybe)
import           System.Random         (randomRIO)
import           Control.DeepSeq       (NFData, force)

import           Torch                 (Tensor, asTensor)

-- | Number of numeric feature columns in the cleaned CSV.
numFeatures :: Int
numFeatures = 52

-- ------------------------------------------------------------------
--  CSV Row representation
-- ------------------------------------------------------------------

-- We parse each row as a generic name-map and then extract
-- the fields we care about.  This avoids having to declare
-- a 52-field record type.

-- | Parse one CSV row (a named record / HashMap) into
--   (label, feature-vector).  Returns Nothing if the row
--   can't be parsed.
parseRow :: Csv.NamedRecord -> Maybe (Text, [Float])
parseRow nr = do
  labelBs <- Csv.lookup nr "Attack Type"     -- ByteString
  let label = labelBs :: Text
  -- All other columns are numeric features; grab them in order.
  -- The column names from the cleaned CSV:
  let featureCols :: [Csv.Name]
      featureCols =
        [ "Destination Port", "Flow Duration"
        , "Total Fwd Packets", "Total Length of Fwd Packets"
        , "Fwd Packet Length Max", "Fwd Packet Length Min"
        , "Fwd Packet Length Mean", "Fwd Packet Length Std"
        , "Bwd Packet Length Max", "Bwd Packet Length Min"
        , "Bwd Packet Length Mean", "Bwd Packet Length Std"
        , "Flow Bytes/s", "Flow Packets/s"
        , "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max", "Flow IAT Min"
        , "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std"
        , "Fwd IAT Max", "Fwd IAT Min"
        , "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std"
        , "Bwd IAT Max", "Bwd IAT Min"
        , "Fwd Header Length", "Bwd Header Length"
        , "Fwd Packets/s", "Bwd Packets/s"
        , "Min Packet Length", "Max Packet Length"
        , "Packet Length Mean", "Packet Length Std"
        , "Packet Length Variance"
        , "FIN Flag Count", "PSH Flag Count", "ACK Flag Count"
        , "Average Packet Size"
        , "Subflow Fwd Bytes"
        , "Init_Win_bytes_forward", "Init_Win_bytes_backward"
        , "act_data_pkt_fwd", "min_seg_size_forward"
        , "Active Mean", "Active Max", "Active Min"
        , "Idle Mean", "Idle Max", "Idle Min"
        ]
  feats <- mapM (parseFloat nr) featureCols
  return (label, feats)

-- | Parse a single numeric cell, handling NaN / Infinity gracefully.
parseFloat :: Csv.NamedRecord -> Csv.Name -> Maybe Float
parseFloat nr col = do
  valText <- Csv.lookup nr col :: Maybe Text
  let cleaned = T.strip valText
  if cleaned == "NaN" || cleaned == "Infinity" || cleaned == "-Infinity"
    then Just 0.0          -- replace bad values with 0
    else case TR.double cleaned of
           Right (d, _) -> Just (realToFrac d)
           Left  _      -> Nothing        -- truly un-parsable → skip row

-- ------------------------------------------------------------------
--  Loading + filtering
-- ------------------------------------------------------------------

-- | Load the CSV, keep only "Normal Traffic" rows, return their
--   feature vectors as a boxed Vector.
loadBenignSamples :: FilePath -> IO (V.Vector [Float])
loadBenignSamples fp = do
  raw <- BL.readFile fp
  case Csv.decodeByName raw of
    Left err   -> error $ "CSV parse error: " ++ err
    Right (_hdr, rows) ->
      let parsed :: [Maybe (Text, [Float])]
          parsed = map parseRow (V.toList rows)
          benign = [ feats
                   | Just (lbl, feats) <- parsed
                   , T.strip lbl == "Normal Traffic"
                   ]
      in do
        let !v = force (V.fromList benign)
        putStrLn $ "Loaded " ++ show (V.length v)
                   ++ " benign samples (" ++ show numFeatures ++ " features each)"
        return v

-- ------------------------------------------------------------------
--  Augmentation (SimCLR needs two views of each sample)
-- ------------------------------------------------------------------

-- | Given a feature vector, produce two augmented copies:
--   1. Add small Gaussian-like jitter  (uniform ±0.05 * |x|)
--   2. Randomly zero-out ~10% of features  (feature masking)
augmentPair :: [Float] -> IO ([Float], [Float])
augmentPair xs = do
  v1 <- augmentOne xs
  v2 <- augmentOne xs
  return (v1, v2)

augmentOne :: [Float] -> IO [Float]
augmentOne = mapM jitterAndMask
  where
    jitterAndMask x = do
      -- jitter: uniform in [-0.05*|x|, +0.05*|x|]
      let mag = abs x * 0.05
      noise <- randomRIO (-mag, mag)
      -- mask: ~10 % chance of zeroing
      coin  <- randomRIO (0 :: Float, 1)
      return $ if coin < 0.1 then 0.0 else x + noise

-- ------------------------------------------------------------------
--  Tensor helper
-- ------------------------------------------------------------------

-- | Stack a list of feature vectors into a 2-D Tensor [batchSize, numFeatures].
batchTensor :: [[Float]] -> Tensor
batchTensor = asTensor
