import pandas as pd
import requests
from datetime import datetime, timedelta

__test__ = False

def test_stooq_symbol(symbol):
    """测试Stooq是否支持某个股票代码"""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)  # 测试最近30天的数据
        
        url = f"https://stooq.com/q/d/l/?s={symbol}&d1={start_date.strftime('%Y%m%d')}&d2={end_date.strftime('%Y%m%d')}&i=d"
        
        df = pd.read_csv(url)
        
        # 检查是否有有效数据
        if len(df) > 0 and 'Date' in df.columns:
            print(f"✅ {symbol}: 支持 - 获取到 {len(df)} 行数据")
            return True
        else:
            print(f"❌ {symbol}: 不支持或无数据")
            return False
            
    except Exception as e:
        print(f"❌ {symbol}: 错误 - {str(e)}")
        return False

def main():
    # 测试一些常见的小公司股票代码
    test_symbols = [
        # 科技类小公司
        'amd.us',      # AMD
        'pltr.us',     # Palantir
        'rblx.us',     # Roblox
        'coin.us',     # Coinbase
        'rivn.us',     # Rivian
        'lcid.us',     # Lucid Motors
        'nio.us',      # 蔚来汽车
        'xpev.us',     # 小鹏汽车
        'li.us',       # 理想汽车

        # 生物医药小公司
        'mrna.us',     # Moderna
        'bntx.us',     # BioNTech
        'nvax.us',     # Novavax

        # 新能源/清洁能源
        'enph.us',     # Enphase Energy
        'sedg.us',     # SolarEdge
        'plug.us',     # Plug Power
        'fcel.us',     # FuelCell Energy

        # 金融科技
        'sq.us',       # Block (Square)
        'pypl.us',     # PayPal
        'hood.us',     # Robinhood

        # 游戏/娱乐
        'u.us',        # Unity Software
        'ttd.us',      # The Trade Desk

        # 其他新兴公司
        'snow.us',     # Snowflake
        'crwd.us',     # CrowdStrike
        'zm.us',       # Zoom
        'docu.us',     # DocuSign
        'roku.us',     # Roku
        'spot.us',     # Spotify
    ]

    print("测试 Stooq 支持的小公司股票代码...")
    print("=" * 50)

    supported_symbols = []
    for symbol in test_symbols:
        if test_stooq_symbol(symbol):
            supported_symbols.append(symbol)

    print("\n" + "=" * 50)
    print(f"总结: 测试了 {len(test_symbols)} 个代码，支持 {len(supported_symbols)} 个")
    print("\n支持的股票代码:")
    for symbol in supported_symbols:
        print(f"  - {symbol}")


if __name__ == "__main__":
    main()
