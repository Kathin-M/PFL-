{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE ScopedTypeVariables #-}

-- | Data pipeline for CICIDS2017.
--   Parses the CSV, filters for "Normal Traffic", and
--   provides augmentation helpers for SimCLR.
module DataPipeline
  ( loadBenignSamples
  , augmentPair
  , batchTensor
  , numFeatures
  ) where

import qualified Data.ByteString.Lazy  as BL
import qualified Data.Csv              as Csv
import qualified Data.Vector           as V
import qualified Data.HashMap.Strict   as HM
import qualified Data.ByteString       as BS
import qualified Data.ByteString.Char8 as BS8
import           Data.Maybe            (mapMaybe)
import           System.Random         (randomRIO)

import           Torch                 (Tensor, asTensor)

-- | Number of numeric feature columns in the cleaned CSV.
numFeatures :: Int
numFeatures = 52

-- ------------------------------------------------------------------
--  Feature column names (in order)
-- ------------------------------------------------------------------

featureColumns :: [BS.ByteString]
featureColumns =
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

-- ------------------------------------------------------------------
--  CSV parsing helpers
-- ------------------------------------------------------------------

-- | Try to parse a ByteString as a Float, returning 0 for NaN/Infinity/errors.
parseFloat :: BS.ByteString -> Float
parseFloat bs =
  let s = BS8.unpack (BS8.strip bs)
  in case s of
       "NaN"       -> 0.0
       "Infinity"  -> 0.0
       "-Infinity" -> 0.0
       ""          -> 0.0
       _           -> case reads s of
                        [(v, "")] -> v
                        _         -> 0.0

-- | Parse one CSV row (NamedRecord = HashMap) into a feature vector.
--   Returns Nothing if the row is not "Normal Traffic".
parseRow :: Csv.NamedRecord -> Maybe [Float]
parseRow nr = do
  labelBs <- HM.lookup "Attack Type" nr
  let label = BS8.strip labelBs
  if label /= "Normal Traffic"
    then Nothing
    else Just $ map (\col -> maybe 0.0 parseFloat (HM.lookup col nr)) featureColumns

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
    Right (_hdr, rows :: V.Vector Csv.NamedRecord) ->
      let benign = mapMaybe parseRow (V.toList rows)
          !v     = V.fromList benign
      in do
        putStrLn $ "Loaded " ++ show (V.length v)
                   ++ " benign samples (" ++ show numFeatures ++ " features each)"
        return v

-- ------------------------------------------------------------------
--  Augmentation (SimCLR needs two views of each sample)
-- ------------------------------------------------------------------

-- | Given a feature vector, produce two augmented copies:
--   1. Add small jitter  (uniform +/- 5% of |x|)
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
      let mag = abs x * 0.05
      noise <- randomRIO (-mag, mag)
      coin  <- randomRIO (0 :: Float, 1)
      return $ if coin < 0.1 then 0.0 else x + noise

-- ------------------------------------------------------------------
--  Tensor helper
-- ------------------------------------------------------------------

-- | Stack a list of feature vectors into a 2-D Tensor [batchSize, numFeatures].
batchTensor :: [[Float]] -> Tensor
batchTensor = asTensor
