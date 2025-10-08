# -*- coding: utf-8 -*-
"""
Simple progress bar utilities for simulation loops.
No external dependencies required.
"""

import sys
import time
from typing import Optional


class ProgressBar:
    """
    Simple progress bar for TTI loops.
    
    Usage:
        with ProgressBar(total=2000, desc="Simulating TTIs") as pbar:
            for i in range(2000):
                # do work
                pbar.update(1)
    """
    
    def __init__(self, total: int, desc: str = "", width: int = 50, 
                 enable: bool = True, update_interval: float = 0.1):
        """
        Args:
            total: Total number of iterations
            desc: Description text to show before progress bar
            width: Width of the progress bar in characters
            enable: Enable/disable progress display
            update_interval: Minimum seconds between updates (reduces overhead)
        """
        self.total = max(1, int(total))
        self.desc = desc
        self.width = width
        self.enable = enable and sys.stdout.isatty()  # Only show if terminal
        self.update_interval = update_interval
        
        self.current = 0
        self.start_time = None
        self.last_update_time = 0
        self.last_print_len = 0
        
    def __enter__(self):
        if self.enable:
            self.start_time = time.time()
            self.last_update_time = self.start_time
            self._print_progress()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            # Force final update
            self._print_progress(force=True)
            sys.stdout.write('\n')
            sys.stdout.flush()
        return False
    
    def update(self, n: int = 1):
        """Update progress by n steps"""
        if not self.enable:
            return
            
        self.current = min(self.total, self.current + n)
        now = time.time()
        
        # Only update display if enough time has passed or we're done
        if (now - self.last_update_time >= self.update_interval) or (self.current >= self.total):
            self._print_progress()
            self.last_update_time = now
    
    def _print_progress(self, force: bool = False):
        """Print the progress bar"""
        if not self.enable and not force:
            return
            
        # Calculate progress
        percent = 100.0 * self.current / self.total
        filled = int(self.width * self.current / self.total)
        bar = '█' * filled + '░' * (self.width - filled)
        
        # Calculate time statistics
        elapsed = time.time() - self.start_time if self.start_time else 0
        
        if self.current > 0:
            rate = self.current / elapsed if elapsed > 0 else 0
            eta_seconds = (self.total - self.current) / rate if rate > 0 else 0
        else:
            rate = 0
            eta_seconds = 0
        
        # Format time strings
        elapsed_str = self._format_time(elapsed)
        eta_str = self._format_time(eta_seconds)
        rate_str = f"{rate:.1f}" if rate > 0 else "?"
        
        # Build progress string
        progress_str = (
            f"\r{self.desc:20s} |{bar}| "
            f"{self.current}/{self.total} "
            f"[{percent:5.1f}%] "
            f"[{elapsed_str}<{eta_str}, {rate_str}it/s]"
        )
        
        # Clear previous line if needed
        if len(progress_str) < self.last_print_len:
            progress_str += ' ' * (self.last_print_len - len(progress_str))
        
        self.last_print_len = len(progress_str)
        
        # Print
        sys.stdout.write(progress_str)
        sys.stdout.flush()
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format seconds as MM:SS or HH:MM:SS"""
        if seconds < 0 or seconds > 86400:  # More than 24 hours
            return "--:--"
        
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"


def progress_wrapper(iterable, total: Optional[int] = None, desc: str = "", 
                     enable: bool = True, **kwargs):
    """
    Wrap an iterable with a progress bar.
    
    Usage:
        for item in progress_wrapper(range(2000), desc="Processing"):
            # do work
            pass
    
    Args:
        iterable: Any iterable object
        total: Total number of items (if not inferable from iterable)
        desc: Description text
        enable: Enable/disable progress display
        **kwargs: Additional arguments for ProgressBar
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            # Iterable doesn't support len()
            total = 0
            enable = False
    
    with ProgressBar(total=total, desc=desc, enable=enable, **kwargs) as pbar:
        for item in iterable:
            yield item
            pbar.update(1)


class SimpleProgressPrinter:
    """
    Simpler milestone-based progress printer for systems where progress bar doesn't work well.
    Prints at milestones (e.g., 10%, 20%, ..., 100%).
    """
    
    def __init__(self, total: int, desc: str = "", milestones: tuple = (10, 25, 50, 75, 90, 100)):
        self.total = max(1, int(total))
        self.desc = desc
        self.milestones = sorted(set(milestones))
        self.current = 0
        self.printed_milestones = set()
        self.start_time = time.time()
        
        # Print start
        print(f"{self.desc}: Starting (total={self.total})...")
    
    def update(self, n: int = 1):
        """Update progress by n steps"""
        self.current = min(self.total, self.current + n)
        percent = 100.0 * self.current / self.total
        
        # Check if we've reached a new milestone
        for milestone in self.milestones:
            if percent >= milestone and milestone not in self.printed_milestones:
                elapsed = time.time() - self.start_time
                rate = self.current / elapsed if elapsed > 0 else 0
                eta = (self.total - self.current) / rate if rate > 0 else 0
                
                print(f"{self.desc}: {milestone}% ({self.current}/{self.total}) - "
                      f"elapsed: {elapsed:.1f}s, rate: {rate:.1f}it/s, "
                      f"ETA: {eta:.1f}s")
                
                self.printed_milestones.add(milestone)
                
                # If we've reached 100%, print final message
                if milestone == 100 or self.current >= self.total:
                    print(f"{self.desc}: Completed in {elapsed:.1f}s")
                    break
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ensure we print 100% if not already printed
        if 100 not in self.printed_milestones and self.current >= self.total:
            elapsed = time.time() - self.start_time
            print(f"{self.desc}: Completed 100% in {elapsed:.1f}s")
        return False
