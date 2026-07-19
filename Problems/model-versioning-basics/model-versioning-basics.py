def promote_model(models):
    """
    Decide which model version to promote to production.
    """
    # Write code here
    max_accuracy = -float("inf")
    min_latency = float("inf")
    max_timestamp = -float("inf")
    ans = None

    for model in models:
        name = model["name"]
        latency = model["latency"]
        accuracy = model["accuracy"]
        timestamp = model["timestamp"]

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            min_latency = latency
            max_timestamp = timestamp
            ans = name

        elif accuracy == max_accuracy:
            if latency < min_latency:
                min_latency = latency
                max_timestamp = timestamp
                ans = name

            elif latency == min_latency and timestamp > max_timestamp:
                max_timestamp = timestamp
                ans = name

    return ans