import numpy as np


def select_views(num_views, min_gap=50, max_gap=150):
    if num_views < 3:
        return None, None
    
    max_gap = min(num_views - 1, max_gap)
    min_gap = min(max(1, min_gap), max_gap)
    
    if max_gap < min_gap:
        max_gap = min_gap
    
    if max_gap <= 0 or min_gap <= 0:
        return None, None
    
    context_gap = np.random.randint(min_gap, max_gap + 1)
    
    if num_views - context_gap <= 0:
        context_gap = max(1, num_views - 1)
    
    context_left = np.random.randint(0, num_views - context_gap)
    context_right = context_left + context_gap
    
    if context_right >= num_views:
        context_right = num_views - 1
        context_left = max(0, context_right - context_gap)
    
    if context_right - context_left <= 1:
        context_left = 0
        context_right = num_views - 1
    
    if context_right - context_left <= 1:
        return None, None
    
    target = np.random.randint(context_left + 1, context_right)
    
    context_indices = [context_left, context_right]
    target_index = target
    
    return context_indices, target_index