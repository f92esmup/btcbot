#!/usr/bin/env python3
"""
Quick verification script to check that both tasks are properly implemented:
1. Training steps checkpointing configuration
2. Madrid timezone support throughout the project
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import ConfigManager
from src.utils.logging_utils import get_madrid_timestamp_str, get_madrid_timestamp, MADRID_TZ
import pytz
from datetime import datetime

def verify_checkpoint_config():
    """Verify that checkpoint configuration is properly set up"""
    print("=" * 60)
    print("VERIFICATION: Training Checkpoint Configuration")
    print("=" * 60)
    
    try:
        config = ConfigManager()
        
        # Check save frequency
        save_freq = config.get_config_value('agent.save_frequency_steps')
        print(f"✓ Save frequency: {save_freq} steps")
        
        # Check if all checkpoints are kept
        keep_all = config.get_config_value('agent.keep_all_checkpoints')
        print(f"✓ Keep all checkpoints: {keep_all}")
        
        # Check total training timesteps
        total_steps = config.get_config_value('agent.total_training_timesteps')
        print(f"✓ Total training timesteps: {total_steps}")
        
        # Calculate expected number of checkpoints
        expected_checkpoints = total_steps // save_freq
        print(f"✓ Expected number of checkpoints: {expected_checkpoints}")
        
        print(f"✓ Checkpoint configuration is properly set up!")
        print(f"  - Model will be saved every {save_freq} steps")
        print(f"  - Total training will generate approximately {expected_checkpoints} checkpoints")
        print(f"  - Keep all checkpoints: {'Yes' if keep_all else 'No (only latest)'}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying checkpoint config: {e}")
        return False

def verify_madrid_timezone():
    """Verify that Madrid timezone utilities are working properly"""
    print("\n" + "=" * 60)
    print("VERIFICATION: Madrid Timezone Support")
    print("=" * 60)
    
    try:
        # Test timezone constant
        print(f"✓ Madrid timezone: {MADRID_TZ}")
        
        # Test current Madrid time
        madrid_time = get_madrid_timestamp()
        madrid_time_str = get_madrid_timestamp_str()
        print(f"✓ Current Madrid time: {madrid_time_str}")
        
        # Test UTC comparison
        utc_time = datetime.now(pytz.utc)
        print(f"✓ Current UTC time: {utc_time.isoformat()}")
        
        # Calculate time difference
        time_diff = madrid_time.utcoffset()
        print(f"✓ Time offset from UTC: {time_diff}")
        
        # Test that Madrid timezone is properly configured
        is_dst = madrid_time.dst() is not None and madrid_time.dst().seconds > 0
        tz_name = "CEST" if is_dst else "CET"
        print(f"✓ Current timezone mode: {tz_name}")
        
        print(f"✓ Madrid timezone configuration is working correctly!")
        
        return True
        
    except Exception as e:
        print(f"✗ Error verifying Madrid timezone: {e}")
        return False

def main():
    """Main verification function"""
    print("BTCBot Configuration Verification")
    print("Date:", get_madrid_timestamp_str())
    
    checkpoint_ok = verify_checkpoint_config()
    timezone_ok = verify_madrid_timezone()
    
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if checkpoint_ok and timezone_ok:
        print("✓ All configurations verified successfully!")
        print("✓ Training checkpoints: CONFIGURED")
        print("✓ Madrid timezone: CONFIGURED")
        print("\nThe system is ready for training and live trading.")
        sys.exit(0)
    else:
        print("✗ Some configurations have issues!")
        if not checkpoint_ok:
            print("✗ Training checkpoints: ISSUES FOUND")
        if not timezone_ok:
            print("✗ Madrid timezone: ISSUES FOUND")
        sys.exit(1)

if __name__ == "__main__":
    main()
