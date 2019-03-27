from functools import reduce

def list_min(series):
    """Finds min value in series
    Args:
        series: 2d list of numbers, e.g: [[1, 2.0, 2.1], [1.0], ...]
    Returns:
        min value, number
    """
    series.size #probably this lines forces lazy-loading of series collection. do not remove  
    return reduce(lambda curr_min, list: curr_min if curr_min < min(list) else min(list), series, series[0][0])

def list_max(series):
    """Finds max value in series
    Args:
        series: 2d list of numbers, e.g: [[1, 2.0, 2.1], [1.0], ...]
    Returns:
        max value, number
    """
    series.size
    return reduce(lambda curr_max, list: curr_max if curr_max > max(list) else max(list), series, series[0][0])

def test_min(series):
    series.size
    return reduce(lambda x, y: x if x < min(y) else min(y), series, series[0][0])