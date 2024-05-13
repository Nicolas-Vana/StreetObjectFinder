def get_bounded_linear_behavior(val, activation, saturation, lower_bound, upper_bound, mode):
    if val <= activation:
        output = upper_bound if mode == 'increasing' else lower_bound
    elif val >= saturation:
        output = lower_bound if mode == 'increasing' else upper_bound
    else:
        linear_portion = (val - activation) / (saturation - activation)
        if mode == 'increasing':
            output = upper_bound - (upper_bound - lower_bound) * linear_portion
        else:  # mode == 'increasing'
            output = lower_bound + (upper_bound - lower_bound) * linear_portion

    return output