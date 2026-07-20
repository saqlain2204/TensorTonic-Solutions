def feature_store_lookup(feature_store, requests, defaults):
    """
    Join offline user features with online request-time features.
    """
    # Write code here
    result = []

    for request in requests:
        user_id = request["user_id"]
        features = defaults.copy()
        features.update(feature_store.get(user_id, {}))
        features.update(request.get("online_features", {}))
        result.append(features)

    return result