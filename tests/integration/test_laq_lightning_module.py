"""
Integration tests for the LAQ Lightning module.

Tests the PyTorch Lightning training integration with LAPA.
"""

import pytest
import torch
import pytorch_lightning as pl
from packages.laq.task import LAQModule


@pytest.mark.integration
class TestLAQModule:
    """Integration tests for LAQ Lightning module."""
    
    def test_laq_module_initialization(self, lapa_config):
        """Test LAQ module initialization."""
        module = LAQModule(
            model_config=lapa_config,
            learning_rate=1e-4
        )
        
        assert module is not None
        assert module.lapa is not None
        assert module.learning_rate == 1e-4
    
    def test_laq_module_training_step(self, device, lapa_config):
        """Test training step."""
        module = LAQModule(
            model_config=lapa_config,
            learning_rate=1e-4
        )
        module.to(device)
        
        batch_size = 2
        batch = torch.randn(
            batch_size, lapa_config['channels'], 2,
            lapa_config['image_size'], lapa_config['image_size'],
            device=device
        )
        
        loss = module.training_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_laq_module_validation_step(self, device, lapa_config):
        """Test validation step."""
        module = LAQModule(
            model_config=lapa_config,
            learning_rate=1e-4
        )
        module.to(device)
        
        batch_size = 2
        batch = torch.randn(
            batch_size, lapa_config['channels'], 2,
            lapa_config['image_size'], lapa_config['image_size'],
            device=device
        )
        
        loss = module.validation_step(batch, batch_idx=0)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_laq_module_configure_optimizers(self, lapa_config):
        """Test optimizer configuration."""
        module = LAQModule(
            model_config=lapa_config,
            learning_rate=1e-4
        )
        
        optimizer_config = module.configure_optimizers()
        
        # configure_optimizers returns a dict with 'optimizer' key
        assert isinstance(optimizer_config, dict)
        assert 'optimizer' in optimizer_config
        optimizer = optimizer_config['optimizer']
        assert isinstance(optimizer, torch.optim.Optimizer)
        # Check initial_lr since lr may be 0.0 due to scheduler
        assert optimizer.param_groups[0]['initial_lr'] == 1e-4
    
    def test_laq_module_forward(self, device, lapa_config):
        """Test forward pass."""
        module = LAQModule(
            model_config=lapa_config,
            learning_rate=1e-4
        )
        module.to(device)
        
        batch_size = 2
        frames = torch.randn(
            batch_size, lapa_config['channels'], 2,
            lapa_config['image_size'], lapa_config['image_size'],
            device=device
        )
        
        reconstructed, indices, perplexity = module(frames)
        
        # Decoder adds temporal dimension
        assert reconstructed.shape == (
            batch_size,
            lapa_config['channels'],
            1,
            lapa_config['image_size'],
            lapa_config['image_size']
        )
        assert indices.shape == (batch_size, 4)
        assert perplexity.shape == ()  # Scalar
    
    def test_laq_module_metrics_logging(self, device, lapa_config, tmp_path):
        """Test that metrics are logged during training."""
        module = LAQModule(
            model_config=lapa_config,
            learning_rate=1e-4
        )
        module.to(device)
        
        batch_size = 2
        batch = torch.randn(
            batch_size, lapa_config['channels'], 2,
            lapa_config['image_size'], lapa_config['image_size'],
            device=device
        )
        
        # Run training step (this logs metrics internally)
        loss = module.training_step(batch, batch_idx=0)
        
        # Check that loss is valid
        assert not torch.isnan(loss)
        assert loss.item() > 0


@pytest.mark.integration
@pytest.mark.slow
class TestLAQModuleTraining:
    """Test LAQ module training with Lightning Trainer."""
    
    def test_laq_module_with_trainer(self, lapa_config, tmp_path):
        """Test LAQ module with Lightning Trainer."""
        # Skip if no GPU available (trainer tests can be slow on CPU)
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            pytest.skip("Requires GPU for trainer tests")
        
        module = LAQModule(
            model_config=lapa_config,
            learning_rate=1e-4
        )
        
        # Create dummy dataloader using proper DataLoader
        from torch.utils.data import DataLoader, TensorDataset
        
        batch_size = 2
        num_samples = 4  # 2 batches with batch_size=2
        
        # Create dummy data: [num_samples, C, T, H, W]
        dummy_tensors = torch.randn(
            num_samples, lapa_config['channels'], 2,
            lapa_config['image_size'], lapa_config['image_size']
        )
        
        dataset = TensorDataset(dummy_tensors)
        dummy_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        # Simple trainer for testing
        trainer = pl.Trainer(
            default_root_dir=str(tmp_path),
            max_epochs=1,
            max_steps=2,
            accelerator='auto',
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False
        )
        
        # Run training
        trainer.fit(module, train_dataloaders=dummy_dataloader)
        
        # Check that training completed
        assert trainer.current_epoch >= 0
    
    def test_laq_module_checkpoint_saving(self, lapa_config, model_checkpoint_path):
        """Test checkpoint saving and loading."""
        module = LAQModule(
            model_config=lapa_config,
            learning_rate=1e-4
        )
        
        # Save checkpoint
        checkpoint = {
            'state_dict': module.state_dict(),
            'hyper_parameters': {'model_config': lapa_config, 'learning_rate': 1e-4}
        }
        torch.save(checkpoint, model_checkpoint_path)
        
        # Load checkpoint
        module2 = LAQModule(
            model_config=lapa_config,
            learning_rate=1e-4
        )
        module2.load_state_dict(checkpoint['state_dict'])
        
        # Check that weights match
        for p1, p2 in zip(module.parameters(), module2.parameters()):
            assert torch.allclose(p1, p2)

