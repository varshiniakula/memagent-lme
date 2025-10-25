"""
Simple Rate Limiter for API Calls
Prevents exceeding API rate limits by tracking and throttling requests.
"""

import time
from collections import deque
from typing import Optional


class RateLimiter:
    """
    Simple rate limiter using sliding window approach.
    
    Usage:
        limiter = RateLimiter(max_requests=15, time_window=60)
        
        # Before each API call:
        limiter.wait_if_needed()
        result = api_call()
    """
    
    def __init__(self, max_requests: int = 15, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed
            time_window: Time window in seconds (default: 60 = 1 minute)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()  # Store timestamps of requests
        
    def wait_if_needed(self):
        """
        Check if we need to wait before making next request.
        Blocks until it's safe to proceed.
        """
        current_time = time.time()
        
        # Remove requests outside the time window (>= to include exactly time_window old)
        while self.requests and current_time - self.requests[0] >= self.time_window:
            self.requests.popleft()
        
        # If we've hit the limit, wait
        if len(self.requests) >= self.max_requests:
            # Calculate how long to wait
            oldest_request = self.requests[0]
            wait_time = self.time_window - (current_time - oldest_request)
            
            if wait_time > 0:
                print(f"   Rate limit: Waiting {wait_time:.1f}s before next request...")
                time.sleep(wait_time + 1.0)  # Add 1 second buffer for safety
                
                # Clean up old requests after waiting
                current_time = time.time()
                while self.requests and current_time - self.requests[0] >= self.time_window:
                    self.requests.popleft()
        
        # Record this request
        self.requests.append(time.time())
    
    def reset(self):
        """Clear all recorded requests."""
        self.requests.clear()
    
    def get_remaining_requests(self) -> int:
        """Get number of requests remaining in current window."""
        current_time = time.time()
        
        # Remove old requests
        while self.requests and current_time - self.requests[0] >= self.time_window:
            self.requests.popleft()
        
        return max(0, self.max_requests - len(self.requests))
    
    def get_stats(self) -> dict:
        """Get current rate limiter statistics."""
        current_time = time.time()
        
        # Remove old requests
        while self.requests and current_time - self.requests[0] >= self.time_window:
            self.requests.popleft()
        
        return {
            'max_requests': self.max_requests,
            'time_window': self.time_window,
            'current_requests': len(self.requests),
            'remaining_requests': self.max_requests - len(self.requests),
            'window_type': f'{self.time_window}s'
        }


# Global rate limiter instance (can be configured)
_global_limiter: Optional[RateLimiter] = None


def get_rate_limiter(max_requests: int = 15, time_window: int = 60) -> RateLimiter:
    """
    Get or create global rate limiter instance.
    
    Args:
        max_requests: Maximum requests per time window
        time_window: Time window in seconds
    
    Returns:
        RateLimiter instance
    """
    global _global_limiter
    
    if _global_limiter is None:
        _global_limiter = RateLimiter(max_requests, time_window)
    
    return _global_limiter


def reset_rate_limiter():
    """Reset the global rate limiter."""
    global _global_limiter
    if _global_limiter:
        _global_limiter.reset()


# Convenience function for direct use
def wait_for_rate_limit(max_requests: int = 15, time_window: int = 60):
    """
    Convenience function to wait if needed before API call.
    
    Args:
        max_requests: Maximum requests per time window
        time_window: Time window in seconds
    
    Usage:
        wait_for_rate_limit()  # Use defaults (15 req/min)
        api_call()
    """
    limiter = get_rate_limiter(max_requests, time_window)
    limiter.wait_if_needed()

