{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE DeriveAnyClass #-}

-- | SimCLR model components:
--     • MLP Encoder  (52 → 128 → 64 → 32)
--     • Projection head (32 → 16)
module SimCLR
  ( EncoderSpec(..)
  , Encoder(..)
  , ProjectionSpec(..)
  , Projection(..)
  , SimCLRModelSpec(..)
  , SimCLRModel(..)
  , encodeForward
  , projectForward
  , simclrForward
  ) where

import GHC.Generics            (Generic)
import Torch

-- ================================================================
--  Encoder: 3-layer MLP  (inputDim → 128 → 64 → latentDim=32)
-- ================================================================

data EncoderSpec = EncoderSpec
  { encInputDim  :: !Int   -- ^ e.g. 52
  , encLatentDim :: !Int   -- ^ e.g. 32
  } deriving (Show, Eq)

data Encoder = Encoder
  { encW1 :: !Linear
  , encW2 :: !Linear
  , encW3 :: !Linear
  } deriving (Generic, Show, Parameterized)

instance Randomizable EncoderSpec Encoder where
  sample EncoderSpec{..} = Encoder
    <$> sample (LinearSpec encInputDim  128)
    <*> sample (LinearSpec 128          64 )
    <*> sample (LinearSpec 64           encLatentDim)

-- | Forward pass through the encoder.
encodeForward :: Encoder -> Tensor -> Tensor
encodeForward Encoder{..} x =
  linear encW3
    . relu
    . linear encW2
    . relu
    . linear encW1
    $ x

-- ================================================================
--  Projection Head: linear  (latentDim → projDim=16)
-- ================================================================

data ProjectionSpec = ProjectionSpec
  { projInputDim  :: !Int  -- ^ e.g. 32
  , projOutputDim :: !Int  -- ^ e.g. 16
  } deriving (Show, Eq)

data Projection = Projection
  { projW :: !Linear
  } deriving (Generic, Show, Parameterized)

instance Randomizable ProjectionSpec Projection where
  sample ProjectionSpec{..} = Projection
    <$> sample (LinearSpec projInputDim projOutputDim)

-- | Forward pass through the projection head.
projectForward :: Projection -> Tensor -> Tensor
projectForward Projection{..} z = linear projW z

-- ================================================================
--  Combined SimCLR model (Encoder + Projection)
-- ================================================================

data SimCLRModelSpec = SimCLRModelSpec
  { smInputDim  :: !Int   -- ^ 52
  , smLatentDim :: !Int   -- ^ 32
  , smProjDim   :: !Int   -- ^ 16
  } deriving (Show, Eq)

data SimCLRModel = SimCLRModel
  { smEncoder    :: !Encoder
  , smProjection :: !Projection
  } deriving (Generic, Show, Parameterized)

instance Randomizable SimCLRModelSpec SimCLRModel where
  sample SimCLRModelSpec{..} = SimCLRModel
    <$> sample (EncoderSpec smInputDim smLatentDim)
    <*> sample (ProjectionSpec smLatentDim smProjDim)

-- | Full forward: input → encoder → projection.
--   Returns the projection-space embedding (used for NT-Xent).
simclrForward :: SimCLRModel -> Tensor -> Tensor
simclrForward SimCLRModel{..} x =
  projectForward smProjection (encodeForward smEncoder x)
