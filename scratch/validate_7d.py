import pandas as pd
import os

os.makedirs('demo', exist_ok=True)

df = pd.read_csv('forecasts/forecast_latest.csv')
df['block_timestamp'] = pd.to_datetime(df['forecast_timestamp'])
df['date'] = df['block_timestamp'].dt.date

# Daily summary
print('=== Daily Forecast Summary June 7-14 ===')
for date, day_df in df.groupby('date'):
    evening = day_df[day_df['block_number'].between(68, 96)]
    solar   = day_df[day_df['block_number'].between(33, 60)]
    peak_block = day_df.loc[day_df['predicted_mcp'].idxmax(), 'block_number']
    print(f'{date}: solar_mean={solar["predicted_mcp"].mean():.0f} evening_max={evening["predicted_mcp"].max():.0f} peak_block={peak_block} peak_mcp={day_df["predicted_mcp"].max():.0f}')

# Overall spike analysis
print(f'\n=== 7-Day Spike Summary ===')
if 'spike_probability' in df.columns:
    spikes = df[df['spike_probability'] > 0.70]
    print(f'Total spike blocks (prob>0.70): {len(spikes)}')
    print(f'Spike block numbers: {sorted(spikes["block_number"].unique().tolist())}')
else:
    print('WARNING: spike_probability missing')

print(f'\nOverall 7d max MCP: {df["predicted_mcp"].max():.0f} Rs')
print(f'Overall 7d min MCP: {df["predicted_mcp"].min():.0f} Rs')
print(f'Overall 7d mean MCP: {df["predicted_mcp"].mean():.0f} Rs')

# Export for your friend
df.to_csv('demo/demo_7d_forecast_june7_14.csv', index=False)
print(f'\nSaved to demo/demo_7d_forecast_june7_14.csv')
