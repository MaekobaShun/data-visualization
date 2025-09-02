from scipy.stats import gaussian_kde
from scipy.stats import skew, kurtosis
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="データ可視化ダッシュボード",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# カスタムCSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        border-bottom: 2px solid #ff7f0e;
        padding-bottom: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file):
    try:
        # エンコーディングを自動判定
        encodings = ['utf-8', 'shift_jis', 'cp932', 'euc-jp']
        for encoding in encodings:
            try:
                df = pd.read_csv(uploaded_file, encoding=encoding, index_col=0)
                return df, None
            except UnicodeDecodeError:
                continue

        # 全て失敗した場合はutf-8で強制読み込み
        df = pd.read_csv(uploaded_file, encoding='utf-8', errors='ignore', index_col=0)
        return df, "エンコーディングの一部に問題がありましたが、読み込みました。"
    except Exception as e:
        return None, f"データ読み込みエラー: {str(e)}"

# 変化しやすいからキャッシュいらない
def get_column_types(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    datetime_cols = []

    # 日付型の可能性があるカラムを検出
    for col in categorical_cols:
        try:
            pd.to_datetime(df[col].dropna().head(100))
            datetime_cols.append(col)
        except:
            pass

    categorical_cols = [col for col in categorical_cols if col not in datetime_cols]

    return numeric_cols, categorical_cols, datetime_cols, bool_cols

# 軽いからキャッシュいらない
def create_summary_stats(df):
    """データ概要統計の生成"""
    summary = {
        '行数': df.shape[0],
        '列数': df.shape[1],
        '数値列数': len(df.select_dtypes(include=[np.number]).columns),
        'カテゴリ列数': len(df.select_dtypes(include=['object', 'category']).columns),
        '欠損値数': df.isnull().sum().sum(),
        '重複行数': df.duplicated().sum()
    }
    return summary

@st.cache_data
# 四分位範囲(IQR)で外れ値を除去
def remove_outliers_iqr(df, cols):
    filtered = df.copy()
    for col in cols:
        # NaNを一時的に除外してIQR計算
        q1 = filtered[col].quantile(0.25)
        q3 = filtered[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        # 外れ値判定（NaNはTrueとして残す）
        mask = (filtered[col] >= lower) & (filtered[col] <= upper)
        mask |= filtered[col].isna()  # NaNをTrueにする
        filtered = filtered[mask]
    return filtered


def main():
    # ヘッダー
    st.markdown('<h1 class="main-header">📊 データ可視化ダッシュボード</h1>', unsafe_allow_html=True)
    #st.title("📊 データ可視化ダッシュボード")
    if st.sidebar.checkbox("📚 機能一覧を表示"):
        st.sidebar.markdown("""
        - 📈 基本統計と概要分析
        - 📊 単変量解析
        - 🔗 相関分析と散布図
        - 📦 外れ値検出
        - 🕰️ 時系列分析
        """)

    st.sidebar.header("📁 データ読み込み")
    uploaded_file = st.sidebar.file_uploader(
        "CSVファイルをアップロード",
        type=["csv"],
        help="UTF-8, Shift-JIS, CP932形式に対応"
    )

    # --------------------------------------
    # ① 先にファイルを読み込む・セッション状態確認
    # --------------------------------------
    # toastのフラグを立てる
    if "uploaded_file" not in st.session_state:
        st.session_state["uploaded_file"] = False

    # ファイルアップロードされたかチェック
    df = None
    if uploaded_file:
        df, error_msg = load_data(uploaded_file)
        if df is None:
            st.error(error_msg)
            st.stop()
        elif error_msg:
            st.warning(error_msg)
        elif not st.session_state["uploaded_file"]:
            st.toast("✅ データを正常に読み込みました！", icon="📁")
            st.session_state["uploaded_file"] = True

    # サンプルデータボタン
    use_demo = st.sidebar.button("🌸 アイリスデータセットを読み込み", help="サンプルデータで試す")
    if use_demo:
        df = px.data.iris()
        st.session_state['sample_data'] = df
        st.rerun()

    # セッションステートからサンプルデータ取得
    if df is None and 'sample_data' in st.session_state:
        df = st.session_state['sample_data']
        st.sidebar.info("📊 デモデータを使用中")

    # --------------------------------------
    # ② どのデータもない → 案内を表示して停止
    # --------------------------------------
    if df is None:
        st.info("👈 左のサイドバーから CSV ファイルをアップロードしてください")
        st.stop()

    # カラム分類
    numeric_cols, categorical_cols, datetime_cols, bool_cols = get_column_types(df)

    st.sidebar.header("📦 外れ値除去")
    remove_outliers = st.sidebar.checkbox("データフレームの外れ値を除去(IQR法)", value=True)

    # toastのフラグを立てる
    if "outlier_toast_shown" not in st.session_state:
        st.session_state["outlier_toast_shown"] = False
    # 使用するdfを切り替える
    df_original = df.copy()
    if remove_outliers:
        before = len(df_original)
        df = remove_outliers_iqr(df_original.copy(), numeric_cols)
        after = len(df)
        st.sidebar.success(f"外れ値除去 ({before} → {after} 行)")
        if not st.session_state["outlier_toast_shown"]:
            st.toast("✅ 外れ値を除去しました！", icon="📦")
            st.session_state["outlier_toast_shown"] = True
    else:
        df = df_original.copy()
        st.sidebar.info("外れ値は除去していません。")

    # サイドバー: フィルタリング機能
    st.sidebar.header("🔍 データフィルタ")
    if categorical_cols:
        filter_col = st.sidebar.selectbox("フィルタ列を選択", ["なし"] + categorical_cols)
        if filter_col != "なし":
            unique_values = df[filter_col].unique()
            selected_values = st.sidebar.multiselect(
                f"{filter_col}の値を選択",
                unique_values,
                default=unique_values
            )
            df = df[df[filter_col].isin(selected_values)]

    # タブ構成
    tabs = st.tabs([
        "📋 データ概要",
        "📊 単変量解析",
        "🔗 相関・関係分析",
        "📦 外れ値分析",
        "🕰️ 時系列分析"
    ])


    with tabs[0]:
        st.markdown('<div class="section-header">📋 データ概要と基本統計</div>', unsafe_allow_html=True)
        # サマリー統計
        summary_stats = create_summary_stats(df)

        summary_text = f"""
        - 📊 **行数**: {summary_stats['行数']}
        - 📈 **列数**: {summary_stats['列数']}
        - 🔢 **数値列数**: {summary_stats['数値列数']}
        - 🔠 **カテゴリ列数**: {summary_stats['カテゴリ列数']}
        - ⚠️ **欠損値数**: {summary_stats['欠損値数']}
        - ♻️ **重複行数**: {summary_stats['重複行数']}
        """
        st.warning(summary_text)

        # データ表示
        st.markdown('<div class="section-header">👀 データプレビュー</div>', unsafe_allow_html=True)
        st.subheader("データデーブル")
        st.dataframe(df)

        # データ型詳細
        st.subheader("データ型詳細")
        dtype_df = pd.DataFrame({
            'カラム名': df.columns,
            'データ型': df.dtypes.astype(str),
            'ユニーク数': df.nunique().values,
            '欠損値数': df.isnull().sum(),
            '欠損率(%)': (df.isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(dtype_df)

        # 基本統計量
        st.subheader("数値データの基本統計量")
        describe_df = df[numeric_cols].describe().T
        describe_df.rename(columns={
            "count" : "合計",
            "mean" : "平均",
            "std" : "標準偏差",
            "min": "最小値",
            "max": "最大値"
        }, inplace=True)
        st.dataframe(describe_df)


    # ===== 単変量解析タブ =====
    with tabs[1]:
        st.markdown('<div class="section-header">📊 単変量解析</div>', unsafe_allow_html=True)
            # タブ構成
        tabs_uni = st.tabs([
            "ブール変数",
            "カテゴリ変数",
            "数値変数",
            "ヴァイオリンプロット"
        ])
        with tabs_uni[0]:
            if len(bool_cols)>0:
                bool_col = st.selectbox("ブール値列を選択", bool_cols, key="bool_dist")
                bool_counts = df[bool_col].value_counts()
                bool_bar = px.bar(
                    x=bool_counts.index,
                    y=bool_counts.values,
                    title=f"{bool_col}の分布",
                    labels={"x": bool_col, "y": "件数"}
                )
                bool_bar.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(bool_bar)
            else:
                st.error("ブール値列が見つかりません")


        with tabs_uni[1]:
            # カテゴリ変数
            if categorical_cols:
                cat_col = st.selectbox("カテゴリ列を選択", categorical_cols, key="cat_dist")
                value_counts = df[cat_col].value_counts()

                col1, col2 = st.columns(2)

                with col1:
                    fig_bar = px.bar(
                        x=value_counts.index,
                        y=value_counts.values,
                        title=f"{cat_col}の分布",
                        labels={"x": cat_col, "y": "件数"}
                    )
                    fig_bar.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig_bar)

                with col2:
                    if len(value_counts) <= 10:
                        fig_pie = px.pie(
                            values=value_counts.values,
                            names=value_counts.index,
                            title=f"{cat_col}の割合"
                        )
                        st.plotly_chart(fig_pie)
                    else:
                        st.info("カテゴリの種類が多いため、円グラフは省略します。")
            else:
                st.info("カテゴリ列が見つかりません。")

        with tabs_uni[2]:
            # 数値変数
            if numeric_cols:
                num_col = st.selectbox("数値列を選択", numeric_cols, key="num_dist")

                with st.expander("詳細設定"):
                    bins = st.slider("ビン数", min_value=2, max_value=30, value=6, key="bins")
                    show_stats_lines = st.checkbox("平均・中央値を表示", value=True, key="stats_lines")
                    show_kde = st.checkbox("密度推定（KDE）を表示", value=False, key="show_kde")

                data = df[num_col].dropna()
                if len(data) < 5:
                    st.error(f"データが少なすぎます（{len(data)}個）。最低5個以上のデータが必要です。")
                else:
                    col1, col2 = st.columns(2)
                    # 左側：ヒストグラム + KDE
                    with col1:
                        fig = go.Figure()
                        # ヒストグラム
                        fig.add_trace(go.Histogram(
                            x=data,
                            name="ヒストグラム",
                            histnorm="density",
                            nbinsx=bins
                        ))

                        # KDE
                        if show_kde and len(data) >= 10:
                            try:
                                kde = gaussian_kde(data)
                                x_vals = np.linspace(data.min(), data.max(), 300)
                                y_vals = kde(x_vals)

                                fig.add_trace(go.Scatter(
                                    x=x_vals,
                                    y=y_vals,
                                    mode="lines",
                                    name="密度推定（KDE）",
                                    line=dict(width=3, color="red"),
                                    hovertemplate='<b>値</b>: %{x:.3f}<br><b>密度</b>: %{y:.4f}<extra></extra>'
                                ))
                            except Exception as e:
                                st.warning(f"KDE計算エラー: {str(e)}")

                        # 統計線
                        if show_stats_lines:
                            mean_val = data.mean()
                            median_val = data.median()
                            fig.add_vline(
                                x=mean_val,
                                line_dash="dash",
                                line_color="green",
                                annotation_text=f"平均: {mean_val:.2f}",
                                annotation_position="top"
                            )
                            fig.add_vline(
                                x=median_val,
                                line_dash="dot",
                                line_color="orange",
                                annotation_text=f"中央値: {median_val:.2f}",
                                annotation_position="bottom"
                            )
                        fig.update_layout(
                            title=f"{num_col}の分布",
                            xaxis_title=num_col,
                            yaxis_title="密度",
                            barmode='overlay',
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                            template='plotly_white'
                        )

                        st.plotly_chart(fig, use_container_width=True)

                    # 右側：ボックスプロット
                    with col2:
                        fig_box = go.Figure()

                        fig_box.add_trace(go.Box(
                            y=data,
                            name=num_col,
                            boxpoints='outliers'
                        ))
                        fig_box.update_layout(
                            title=f"{num_col}のボックスプロット",
                            yaxis_title=num_col,
                            template='plotly_white'
                        )
                        st.plotly_chart(fig_box, use_container_width=True)

                # 統計情報サマリー
                describe_stats = data.describe().to_frame(name='値')  # Series → DataFrameへ
                describe_stats.loc['skewness（歪度）'] = skew(data)
                describe_stats.loc['kurtosis（尖度）'] = kurtosis(data)

                st.info(f"""
                **統計サマリー:**
                - **平均**: {data.mean():.3f}
                - **中央値**: {data.median():.3f}
                - **標準偏差**: {data.std():.3f}
                - **最小値**: {data.min():.3f}
                - **最大値**: {data.max():.3f}
                - **データ数**: {len(data)}
                - **歪度（skewness）**: {skew(data):.3f}
                - **尖度（kurtosis）**: {kurtosis(data):.3f}
                """)

            else:
                st.info("数値列が見つかりません。")


        with tabs_uni[3]:
            # ヴァイオリンプロットタブ
            if numeric_cols:
                # 変数選択
                violin_col = st.selectbox("数値列", numeric_cols, key="violin_col")

                # 詳細設定
                with st.expander("詳細設定"):
                    group_col = None
                    if categorical_cols:
                        use_group = st.checkbox("グループ化", key="violin_group")
                        if use_group:
                            group_col = st.selectbox("グループ列", categorical_cols, key="violin_group_col")
                    show_box = st.checkbox("ボックスプロットを重ねて表示", value=True, key="violin_box")
                    show_points = st.checkbox("データポイントを表示", value=False, key="violin_points")
                    violin_side = st.selectbox("表示方向", ["both", "positive", "negative"], key="violin_side")

                col1, col2 = st.columns(2)

                # 左側：ヴァイオリンプロット
                with col1:
                    fig_violin = px.violin(
                        df,
                        y=violin_col,
                        x=group_col,
                        box=show_box,
                        points="all" if show_points else False,
                        title=f"{violin_col}のヴァイオリンプロット",
                        template='plotly_white'
                    )

                    # 表示方向の設定
                    if violin_side != "both":
                        fig_violin.update_traces(side=violin_side)

                    st.plotly_chart(fig_violin, use_container_width=True)

                # 右側：統計情報
                with col2:
                    if group_col:
                        st.subheader("📊 グループ別統計")

                        group_stats = df.groupby(group_col)[violin_col].agg([
                            'count', 'mean', 'median', 'std', 'min', 'max'
                        ]).round(3)

                        group_stats.columns = ['データ数', '平均', '中央値', '標準偏差', '最小値', '最大値']
                        st.dataframe(group_stats, use_container_width=True)

                        # 分散分析（ANOVA）
                        try:
                            from scipy.stats import f_oneway
                            groups = [group[violin_col].dropna() for name, group in df.groupby(group_col)]
                            if len(groups) >= 2 and all(len(g) > 0 for g in groups):
                                f_stat, p_value = f_oneway(*groups)

                                st.info(f"""
                                **分散分析（ANOVA）結果:**
                                - **F統計量**: {f_stat:.3f}
                                - **p値**: {p_value:.6f}
                                - **有意差**: {'あり' if p_value < 0.05 else 'なし'}（α=0.05）
                                """)
                        except ImportError:
                            st.info("scipy がインストールされていないため、ANOVA分析は表示できません。")
                    else:
                        data = df[violin_col].dropna()
                        st.warning(f"""
                        ### 📊 基本統計
                        - **データ数**: {len(data)}
                        - **平均**: {data.mean():.3f}
                        - **中央値**: {data.median():.3f}
                        - **標準偏差**: {data.std():.3f}
                        - **最小値**: {data.min():.3f}
                        - **第1四分位数**: {data.quantile(0.25):.3f}
                        - **第3四分位数**: {data.quantile(0.75):.3f}
                        - **最大値**: {data.max():.3f}
                        """)
            else:
                st.info("ヴァイオリンプロットには数値列が必要です。")

    # ----- タブ: 相関・関係分析 -----
    with tabs[2]:
        st.markdown(
            '<div class="section-header">🔗 相関・関係分析</div>',
            unsafe_allow_html=True
        )

        if len(numeric_cols) < 2:
            st.info("相関分析には2つ以上の数値列が必要です。")
        else:
            corr_tabs = st.tabs(["散布図分析", "相関行列", "ペアプロット"])

            # --- 散布図分析タブ ---
            with corr_tabs[0]:

                # 詳細設定
                with st.expander("詳細設定"):
                    x_var = st.selectbox("X軸変数", numeric_cols, key="scatter_x")
                    y_default = 1 if len(numeric_cols) > 1 else 0
                    y_var = st.selectbox("Y軸変数", numeric_cols, index=y_default, key="scatter_y")

                    show_trend = st.checkbox("回帰線を表示", value=True, key="show_trend")
                    color_var = None
                    if categorical_cols:
                        color_var = st.selectbox(
                            "色分け変数", categorical_cols, key="color_var"
                        )
                        show_trend = False

                # プロットと統計
                plot_col, stats_col = st.columns([1, 1])
                with plot_col:
                    fig = px.scatter(
                        df,
                        x=x_var,
                        y=y_var,
                        color=color_var,
                        trendline="ols" if show_trend and color_var is None else None,
                        title=f"{x_var} vs {y_var}",
                        template="plotly_white"
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with stats_col:
                    # 相関係数
                    corr = df[x_var].corr(df[y_var])
                    valid = df[[x_var, y_var]].dropna()
                    count = len(valid)
                    strength = (
                        "非常に強い" if abs(corr) >= 0.8 else
                        "強い"   if abs(corr) >= 0.6 else
                        "中程度" if abs(corr) >= 0.4 else
                        "弱い"   if abs(corr) >= 0.2 else
                        "非常に弱い"
                    )
                    direction = (
                        "正の相関" if corr > 0.1 else
                        "負の相関" if corr < -0.1 else
                        "ほぼ無相関"
                    )

                    st.warning(f"""
                    ### 🔗 相関統計
                    - 相関係数: {corr:.3f}
                    - データ数: {count}
                    - 強さ: {strength}
                    - 方向: {direction}
                    """)

                    stat_info = f"""
                    ### 📊 変数の基本統計量
                    **{x_var}**
                    - 平均: {df[x_var].mean():.3f}
                    - 標準偏差: {df[x_var].std():.3f}
                    - 最小値: {df[x_var].min():.3f}
                    - 最大値: {df[x_var].max():.3f}

                    **{y_var}**
                    - 平均: {df[y_var].mean():.3f}
                    - 標準偏差: {df[y_var].std():.3f}
                    - 最小値: {df[y_var].min():.3f}
                    - 最大値: {df[y_var].max():.3f}
                    """
                    st.info(stat_info)

            # --- 相関行列タブ ---
            with corr_tabs[1]:
                corr_mtx = df[numeric_cols].corr()
                heat_col, list_col = st.columns([1, 1])

                with heat_col:
                    heat_fig = px.imshow(
                        corr_mtx,
                        text_auto=".3f",
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="数値変数間の相関行列",
                        template="plotly_white"
                    )
                    heat_fig.update_layout(xaxis=dict(tickangle=45), height=500)
                    st.plotly_chart(heat_fig, use_container_width=True)

                with list_col:
                    # 相関行列の統計（info形式で表示）
                    upper = corr_mtx.where(np.triu(np.ones(corr_mtx.shape), k=1).astype(bool))
                    all_vals = upper.stack().dropna()
                    summary_text = f"""
                    ### 📊 相関行列の統計
                    - **変数ペア数**: {len(all_vals)}
                    - **平均相関係数**: {all_vals.mean():.3f}
                    - **最大相関係数**: {all_vals.max():.3f}
                    - **最小相関係数**: {all_vals.min():.3f}
                    """
                    st.warning(summary_text)

                    st.subheader("📋 強い相関の組み合わせ")
                    thresh = st.slider("相関の閾値", 0.1, 0.9, 0.3, 0.1, key="corr_threshold")
                    pairs = []
                    cols = corr_mtx.columns
                    for i in range(len(cols)):
                        for j in range(i+1, len(cols)):
                            val = corr_mtx.iloc[i, j]
                            if abs(val) >= thresh:
                                pairs.append((cols[i], cols[j], val))
                    if pairs:
                        df_pairs = pd.DataFrame(
                            sorted(pairs, key=lambda x: abs(x[2]), reverse=True),
                            columns=["変数1", "変数2", "相関係数"]
                        )
                        st.dataframe(df_pairs.drop(columns=["相関係数"]), use_container_width=True)

                        summary_corr_text = f"""
                        ### 🔗 相関サマリー
                        - **閾値超えの組み合わせ数**: {len(pairs)}
                        - **最大相関係数**: {max(pairs, key=lambda x: abs(x[2]))[2]:.3f}
                        - **平均相関係数**: {np.mean([v for *_, v in pairs]):.3f}
                        """
                        st.info(summary_corr_text)
                    else:
                        st.info(f"相関係数が{thresh}以上の組み合わせがありません。")

            # ===== ペアプロットタブ =====
            with corr_tabs[2]:
                if len(numeric_cols) < 2:
                    st.info("ペアプロットには2つ以上の数値列が必要です。")
                else:
                    # 列選択（最大6列推奨）
                    max_cols = min(6, len(numeric_cols))
                    pair_cols = st.multiselect(
                        "ペアプロット対象列（最大6列推奨）",
                        numeric_cols,
                        default=numeric_cols[:max_cols],
                        key="pair_cols"
                    )

                    # 色分けオプション
                    color_pair = None
                    if categorical_cols:
                        use_color_pair = st.checkbox("色分け", key="pair_color")
                        if use_color_pair:
                            color_pair = st.selectbox("色分け列", categorical_cols, key="pair_color_col")

                    # サンプルサイズ指定（0で全件）
                    sample_size = st.number_input(
                        "サンプル数（0: 全件）",
                        min_value=0,
                        max_value=len(df),
                        value=0,
                        step=100,
                        help="0を指定すると全データを使用します"
                    )

                    # 選択列数チェック
                    if len(pair_cols) < 2:
                        st.info("2つ以上の列を選択してください。")
                    elif len(pair_cols) > 6:
                        st.warning("ペアプロットは6列以下で実行してください。パフォーマンスに影響する可能性があります。")
                    else:
                        # プロット生成ボタン
                        if st.button("ペアプロットを生成", key="generate_pair"):
                            col1, col2 = st.columns([2, 1])

                            # 左：散布図行列（ペアプロット）
                            with col1:
                                plot_cols = pair_cols + ([color_pair] if color_pair else [])
                                data_to_plot = df[plot_cols]
                                if sample_size > 0 and sample_size < len(data_to_plot):
                                    data_to_plot = data_to_plot.sample(n=sample_size, random_state=42)

                                fig_pair = px.scatter_matrix(
                                    data_to_plot,
                                    dimensions=pair_cols,
                                    color=color_pair,
                                    title="ペアプロット（散布図行列）",
                                    template="plotly_white"
                                )
                                fig_pair.update_layout(height=600)
                                st.plotly_chart(fig_pair, use_container_width=True)

                            # 右：相関行列と強い相関ペア
                            with col2:
                                st.subheader("📊 相関行列")
                                corr_matrix = df[pair_cols].corr()

                                fig_corr = px.imshow(
                                    corr_matrix,
                                    text_auto=".2f",
                                    aspect="auto",
                                    color_continuous_scale="RdBu_r",
                                    title="相関行列",
                                    template="plotly_white"
                                )
                                fig_corr.update_layout(height=400)
                                st.plotly_chart(fig_corr, use_container_width=True)

                                st.subheader("📋 強い相関 (|r| ≥ 0.5)")
                                strong_corrs = [
                                    {
                                        "変数ペア": f"{pair_cols[i]} - {pair_cols[j]}",
                                        "相関係数": f"{corr_matrix.iloc[i, j]:.3f}"
                                    }
                                    for i in range(len(pair_cols))
                                    for j in range(i + 1, len(pair_cols))
                                    if abs(corr_matrix.iloc[i, j]) >= 0.5
                                ]

                                if strong_corrs:
                                    st.dataframe(pd.DataFrame(strong_corrs), use_container_width=True)
                                else:
                                    st.info("相関係数0.5以上の組み合わせはありませんでした。")


    # ===== 外れ値分析タブ =====
    with tabs[3]:
        st.markdown('<div class="section-header">📦 外れ値分析</div>', unsafe_allow_html=True)
        target_col = st.selectbox("対象列", numeric_cols, key="target_col")
        threshold = st.slider("Z-Scoreしきい値", 1.0, 4.0, 3.0, 0.1)
        if numeric_cols:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 箱ひげ図")
                fig_box = px.box(
                    df, y=target_col,
                    title=f"{target_col}の箱ひげ図"
                )
                st.plotly_chart(fig_box)

                # 外れ値の統計
                Q1 = df[target_col].quantile(0.25)
                Q3 = df[target_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[target_col] < lower_bound) | (df[target_col] > upper_bound)]
                st.info(f"IQR法による外れ値: {len(outliers)}件 ({len(outliers)/len(df)*100:.1f}%)")

            with col2:
                st.subheader("📏 Z-Score分析")

                z_scores = np.abs((df[target_col] - df[target_col].mean()) / df[target_col].std())
                outliers_z = df[z_scores > threshold]

                fig_z = go.Figure()
                fig_z.add_trace(go.Scatter(
                    x=df.index, y=z_scores,
                    mode='markers',
                    name='Z-Score',
                    marker=dict(
                        color=np.where(z_scores > threshold, 'red', 'blue'),
                        size=6
                    )
                ))
                fig_z.add_hline(y=threshold, line_dash="dash", line_color="red")
                fig_z.update_layout(
                    title=f"{target_col}のZ-Score",
                    xaxis_title="インデックス",
                    yaxis_title="Z-Score"
                )
                st.plotly_chart(fig_z)
                st.info(f"Z-Score法による外れ値: {len(outliers_z)}件 ({len(outliers_z)/len(df)*100:.1f}%)")
        else:
            st.warning("外れ値分析には数値列が必要です。")


    #===== 時系列分析タブ =====
    with tabs[4]:
        st.error("開発中・・・")
        st.header("🕰️ 時系列分析")


if __name__ == "__main__":
    main()