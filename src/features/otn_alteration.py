"""OTN-specific features alteration functions and bundles.

/!\\ `Output` corresponds to a service's input node, `Input` to a service's output node.
"""


def add_network_gains(period_df, original_treatment):
    """Adds total and laser gain features, keeping or not the original inputs.

    G ~ log(Po/Pi): gains are proportional to the differences between the output and input log-powers.
    """
    altered_df = period_df.copy()
    for signal in ['Total', 'Laser']:
        altered_df[f'{signal}Gain'] = period_df[f'{signal}InputPower'] - period_df[f'{signal}OutputPower']
    # drop the original input features if specified
    if original_treatment == 'drop':
        altered_df = altered_df.drop(period_df.columns, axis=1)
    return altered_df


# list of features alteration bundles relevant to otn data
OTN_BUNDLES = [
    # bundle #0 - add network gain information
    {
        'all': 'identity.gains_keep'
    },
    # bundle #1 - option #2 (add few window statistics, rolling 3 days, keep original)
    {
        'all': 'identity.gains_keep.window_2_rolling_keep_3d'
    },
    # bundle #2 - option #3 (add all relevant window features in 3 domains)
    {
        'all': 'identity.gains_keep.window_3_rolling_keep_3d'
    },
    # bundle #3 - option #4 (add a restricted amount of window features for efficiency)
    {
        'all': 'identity.gains_keep.window_4_rolling_keep_3d'
    }
]
