{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE FunctionalDependencies #-}
{-# LANGUAGE OverloadedRecordDot #-}
{-# LANGUAGE DuplicateRecordFields #-}

-- | SimCLR model components:
--     * MLP Encoder  (52 -> 128 -> 64 -> 32)
--     * Projection head (32 -> 16)
module SimCLR
  ( Encoder(..)
  , Projection(..)
  , SimCLRModel(..)
  , initSimCLR
  , encodeForward
  , projectForward
  , simclrForward
  ) where

import GHC.Generics            (Generic)
import Torch

-- ================================================================
--  Encoder: 3-layer MLP  (inputDim -> 128 -> 64 -> latentDim=32)
-- ================================================================

data Encoder = Encoder
  { encW1 :: Linear
  , encW2 :: Linear
  , encW3 :: Linear
  } deriving (Generic, Show, Parameterized)

initEncoder :: Int -> Int -> IO Encoder
initEncoder inputDim latentDim = do
  w1 <- sample $ LinearSpec { in_features = inputDim,  out_features = 128 }
  w2 <- sample $ LinearSpec { in_features = 128,       out_features = 64  }
  w3 <- sample $ LinearSpec { in_features = 64,        out_features = latentDim }
  return $ Encoder w1 w2 w3

-- | Forward pass through the encoder.
encodeForward :: Encoder -> Tensor -> Tensor
encodeForward Encoder{..} x =
    relu (linear encW3 (relu (linear encW2 (relu (linear encW1 x)))))

-- ================================================================
--  Projection Head: linear  (latentDim -> projDim=16)
-- ================================================================

data Projection = Projection
  { projW :: Linear
  } deriving (Generic, Show, Parameterized)

initProjection :: Int -> Int -> IO Projection
initProjection inDim outDim = do
  w <- sample $ LinearSpec { in_features = inDim, out_features = outDim }
  return $ Projection w

-- | Forward pass through the projection head.
projectForward :: Projection -> Tensor -> Tensor
projectForward Projection{..} z = linear projW z

-- ================================================================
--  Combined SimCLR model (Encoder + Projection)
-- ================================================================

data SimCLRModel = SimCLRModel
  { smEncoder    :: Encoder
  , smProjection :: Projection
  } deriving (Generic, Show, Parameterized)

-- | Create a fresh randomly-initialised SimCLR model.
initSimCLR :: Int -> Int -> Int -> IO SimCLRModel
initSimCLR inputDim latentDim projDim = do
  enc  <- initEncoder inputDim latentDim
  proj <- initProjection latentDim projDim
  return $ SimCLRModel enc proj

-- | Full forward: input -> encoder -> projection.
--   Returns the projection-space embedding (used for NT-Xent).
simclrForward :: SimCLRModel -> Tensor -> Tensor
simclrForward SimCLRModel{..} x =
  projectForward smProjection (encodeForward smEncoder x)
