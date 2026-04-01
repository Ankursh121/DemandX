def get_best_discount(df):
    return df.groupby("discount")["sales"].mean().idxmax()

def generate_insight(df):
    best_discount = get_best_discount(df)

    return f"""
    🔥 Best Discount: {best_discount}%  
    📈 Higher marketing + moderate discount boosts sales  
    🎯 Recommendation: Keep discount around {best_discount}%
    """