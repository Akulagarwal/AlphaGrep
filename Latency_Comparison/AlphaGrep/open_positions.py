def print_position_exposure(response):
    try:
        data = response.json() if hasattr(response, 'json') else response
    except (AttributeError, ValueError) as e:
        print(f"Error parsing response: {e}")
        return

    asset_positions = data.get("assetPositions", [])
    
    if not asset_positions:
        print("No open positions.")
        return

    print("Open Positions:")
    print("-" * 50)
    net_exposure = 0.0

    for asset in asset_positions:
        position = asset.get("position", {})
        coin = position.get("coin", "Unknown")
        try:
            szi = float(position.get("szi", 0))
            position_value = float(position.get("positionValue", 0))
        except (ValueError, TypeError):
            print(f"Invalid data for {coin}: Skipping")
            continue
        signed_value = -position_value if szi < 0 else position_value
        net_exposure += signed_value
        position_type = "Short" if szi < 0 else "Long"
        print(f"Instrument: {coin}")
        print(f"Size: {szi} {coin} ({position_type})")
        print(f"Position Value: ${position_value:.2f}")
        print("-" * 50)

    try:
        gross_exposure = float(data.get("crossMarginSummary", {}).get("totalNtlPos", 0))
    except (ValueError, TypeError):
        print("Error retrieving gross exposure")
        return

    print("Exposure Summary:")
    print("-" * 50)
    print(f"Gross Exposure: ${gross_exposure:.2f}")
    print(f"Net Exposure: ${net_exposure:.2f}")