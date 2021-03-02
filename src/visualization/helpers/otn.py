"""OTN-specific visualization helpers.
"""
# performance can only be reported globally here
METRICS_COLORS = 'blue'


def get_service_locations(service_no, datasets_info):
    """Returns the periods locations for the provided service within the datasets.

    The "locations" of a service are defined as the dataset name and dataset position
    of each of its periods.

    Args:
        service_no (int): service number (i.e. name) to show (the same as its file name).
        datasets_info (dict): of the form {`set_name`: `periods_info`};
            with `periods_info` a list of the form `[file_name, trace_type, period_rank]`
            for each period of the set.

    Returns:
        list: the list of `(set_name, set_position)` of the service periods, with
            such tuples ranked chronologically as the periods occur in the service.
    """
    service_locations = []
    for set_name in datasets_info:
        for i, (file_name, service_type, period_rank) in enumerate(datasets_info[set_name]):
            if file_name == service_no:
                service_locations.append((set_name, i, period_rank))
    return [t[:2] for t in sorted(service_locations, key=lambda t: t[2])]
