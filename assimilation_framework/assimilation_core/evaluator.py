
def evaluate_from_metric(xb, xa, gt, metric, metric_agent):
    if metric == "bg_wrmse":
        return metric_agent.WRMSE(xb, gt, None, None, 1)
        
    elif metric == "bg_bias":
        return metric_agent.Bias(xb, gt, None, None, 1)

    elif metric == "ana_wrmse":
        return metric_agent.WRMSE(xa, gt, None, None, 1)
        
    elif metric == "ana_bias":
        return metric_agent.Bias(xa, gt, None, None, 1)

