# =========================================================================
# 可視化的自動跟隨更新 (對標 SPY) 
# =========================================================================

if len(historical_dates) > 0 and not failed_due_to_margin:
    import yfinance as yf
    print("Fetching SPY benchmark...")
    spy_df = yf.download("SPY", start=str(historical_dates[0].date()), end=str(historical_dates[-1].date()), progress=False)
    spy_close = spy_df['Close'].squeeze()
    spy_close.index = pd.to_datetime(spy_close.index).tz_localize(None)

    results_df = pd.DataFrame({'Date': historical_dates, 'Agent_NAV': historical_portfolio_nav}).set_index('Date')
    
    spy_aligned = spy_close.reindex(results_df.index).ffill()
    spy_initial = spy_aligned.iloc[0]
    results_df['SPY_NAV'] = (spy_aligned / spy_initial) * INITIAL_CAPITAL
    
    results_df['Agent_Peak'] = results_df['Agent_NAV'].cummax()
    results_df['Drawdown'] = (results_df['Agent_NAV'] - results_df['Agent_Peak']) / results_df['Agent_Peak']
    
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, row_heights=[0.7, 0.3],
                        subplot_titles=('Rolling Tranches Portfolio NAV vs S&P 500', 'Strategy Drawdown Analysis (%)'))

    fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Agent_NAV'], name='Portfolio NAV ($)', line=dict(color='indigo', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df['SPY_NAV'], name='SPY Benchmark ($)', line=dict(color='gray', dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=results_df.index, y=results_df['Drawdown']*100, name='Drawdown (%)', fill='tozeroy', line=dict(color='red', width=1)), row=2, col=1)

    fig.update_layout(title=f'Rolling Tranches Strategy (Initial: ${INITIAL_CAPITAL})', height=800, template='plotly_white')
    fig.show()

    total_return = (results_df['Agent_NAV'].iloc[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    
    
    print("\n📊 【滾動梯隊資金管理 - 最終總結結算】 📊")
    print(f"🏁 最終資金餘額: ${results_df['Agent_NAV'].iloc[-1]:,.2f}")
    print(f"📈 累計淨報酬率: {total_return * 100:.2f}%")
    print(f"📉 最大回撤 (MDD): {results_df['Drawdown'].min() * 100:.2f}%")
    print(f"🔄 成功執行退歸的梯隊: {completed_tranches_count} 梯")
    print(f"🛒 全期間總交易次數: {sum(trade_frequencies)} 次\n")