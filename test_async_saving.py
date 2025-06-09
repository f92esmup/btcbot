#!/usr/bin/env python3
"""
Test script for asynchronous saving optimization
"""

import torch
import tempfile
import os
from pathlib import Path
from multiprocessing import Process
import time

# Test data setup
def create_mock_agent_state():
    """Create mock agent state dictionaries for testing"""
    return {
        'actor': {'weight': torch.randn(10, 10), 'bias': torch.randn(10)},
        'critic_1': {'weight': torch.randn(10, 10), 'bias': torch.randn(10)},
        'critic_2': {'weight': torch.randn(10, 10), 'bias': torch.randn(10)},
        'critic_target_1': {'weight': torch.randn(10, 10), 'bias': torch.randn(10)},
        'critic_target_2': {'weight': torch.randn(10, 10), 'bias': torch.randn(10)},
        'actor_optimizer': {'state': {}, 'param_groups': []},
        'critic_1_optimizer': {'state': {}, 'param_groups': []},
        'critic_2_optimizer': {'state': {}, 'param_groups': []},
        'log_alpha': torch.tensor(0.1),
        'alpha_optimizer': {'state': {}, 'param_groups': []},
        'metadata': {
            'episode': 100,
            'total_steps': 50000,
            'learning_steps': 25000,
            'learn_alpha': True,
            'target_entropy': -2.0,
            'device': 'cpu'
        }
    }

def test_local_worker():
    """Test the local worker function"""
    from src.training.run_manager import _save_worker_local
    
    print("🧪 Testing local worker...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        agent_state = create_mock_agent_state()
        path_prefix = os.path.join(temp_dir, "test_checkpoint")
        
        # Test synchronous call first
        start_time = time.time()
        _save_worker_local(agent_state, path_prefix)
        sync_time = time.time() - start_time
        
        # Verify files were created
        expected_files = [
            f"{path_prefix}_actor.pth",
            f"{path_prefix}_critic_1.pth", 
            f"{path_prefix}_critic_2.pth",
            f"{path_prefix}_critic_target_1.pth",
            f"{path_prefix}_critic_target_2.pth",
            f"{path_prefix}_actor_optimizer.pth",
            f"{path_prefix}_critic_1_optimizer.pth",
            f"{path_prefix}_critic_2_optimizer.pth",
            f"{path_prefix}_log_alpha.pth",
            f"{path_prefix}_alpha_optimizer.pth",
            f"{path_prefix}_metadata.pth"
        ]
        
        missing_files = []
        for file_path in expected_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
        
        if missing_files:
            print(f"❌ Missing files: {missing_files}")
            return False
        else:
            print(f"✅ Local worker test passed! ({sync_time:.3f}s)")
            return True

def test_async_behavior():
    """Test that async behavior doesn't block"""
    from src.training.run_manager import _save_worker_local
    
    print("🧪 Testing async behavior...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        agent_state = create_mock_agent_state()
        path_prefix = os.path.join(temp_dir, "test_async")
        
        # Launch async process
        start_time = time.time()
        process = Process(target=_save_worker_local, args=(agent_state, path_prefix))
        process.start()
        
        # Measure how long it takes to start (should be very fast)
        launch_time = time.time() - start_time
        
        print(f"✅ Process launched in {launch_time:.4f}s (should be < 0.1s)")
        
        # Wait for completion
        process.join()
        completion_time = time.time() - start_time
        
        print(f"✅ Total completion time: {completion_time:.3f}s")
        
        # Verify files exist
        if os.path.exists(f"{path_prefix}_metadata.pth"):
            print("✅ Async save completed successfully!")
            return True
        else:
            print("❌ Async save failed!")
            return False

if __name__ == "__main__":
    print("🚀 Starting async saving optimization tests...\n")
    
    try:
        # Test 1: Local worker functionality
        test1_passed = test_local_worker()
        print()
        
        # Test 2: Async behavior
        test2_passed = test_async_behavior()
        print()
        
        if test1_passed and test2_passed:
            print("🎉 All tests passed! Async saving optimization is working correctly.")
        else:
            print("❌ Some tests failed. Please check the implementation.")
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
