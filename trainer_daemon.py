import os
import sys
import time
import subprocess
import datetime
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] TRAINER_DAEMON: %(message)s")
log = logging.getLogger("trainer_daemon")

def run_nightly_training():
    log.info("Triggering nightly MARL training sequence...")
    bat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_nightly.bat")
    
    if not os.path.exists(bat_path):
        log.error(f"Cannot find {bat_path}")
        return
        
    try:
        # Run the batch script
        result = subprocess.run(
            [bat_path], 
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            log.info("Nightly training completed successfully.")
        else:
            log.error(f"Nightly training failed with return code: {result.returncode}")
            log.error(f"STDOUT: {result.stdout}")
            log.error(f"STDERR: {result.stderr}")
    except Exception as e:
        log.error(f"Failed to execute training sequence: {e}")

def main():
    log.info("Trainer Daemon initialized. Waiting for 00:00 (Midnight)...")
    
    # Check if we were passed a flag to test instantly
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        log.info("Test mode enabled. Running training sequence immediately...")
        run_nightly_training()
        log.info("Test complete. Exiting.")
        return

    while True:
        now = datetime.datetime.now()
        
        # Check if it is currently exactly 00:00 (midnight)
        if now.hour == 0 and now.minute == 0:
            run_nightly_training()
            
            # Sleep for 60 seconds to ensure we don't trigger again within the same minute
            log.info("Sleeping for 61 seconds to prevent double-trigger...")
            time.sleep(61)
            
        # Sleep for a short duration before checking again
        time.sleep(15)

if __name__ == "__main__":
    main()
