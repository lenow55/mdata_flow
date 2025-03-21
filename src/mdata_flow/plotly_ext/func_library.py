import numpy as np
import pandas as pd
from numpy.typing import NDArray
from plotly import colors
from plotly import graph_objects as go
from scipy import stats
from typing_extensions import Any


def plot_qq(
    y_test: "pd.Series[Any] | NDArray[Any]",
    y_pred: "pd.Series[Any] | NDArray[Any]",
    plot_size: tuple[int, int] = (800, 600),
):
    # Вычисляем остатки
    residuals = y_test - y_pred

    # Создаем график
    fig = go.Figure()

    qq = stats.probplot(residuals, dist="norm")
    x = np.array([qq[0][0][0], qq[0][0][-1]])

    # Добавляем точки QQ-графика
    _ = fig.add_trace(  ## pyright: ignore[reportUnknownMemberType]
        go.Scatter(
            x=qq[0][0],
            y=qq[0][1],
            mode="markers",
            name="QQ Plot",
            marker=dict(color="blue", opacity=0.4),
        )
    )

    # Добавляем линию 45 градусов (для идеального нормального распределения)
    _ = fig.add_trace(  ## pyright: ignore[reportUnknownMemberType]
        go.Scatter(
            x=x,
            y=qq[1][1] + qq[1][0] * x,  ##pyright: ignore[reportOperatorIssue]
            mode="lines",
            name="Ideal Line",
            line=dict(color="red", dash="dash"),
        )
    )

    # Настроить макет
    _ = fig.update_layout(  ## pyright: ignore[reportUnknownMemberType]
        title="QQ Plot",
        xaxis_title="Theoretical Quantiles",
        yaxis_title="Sample Quantiles",
        template="plotly_white",
        width=plot_size[0],
        height=plot_size[1],
        margin=dict(l=100, r=50, t=50, b=50),
    )

    return fig


def plot_correlation_matrix(df: pd.DataFrame, plot_size: tuple[int, int] = (800, 800)):
    # Calculate the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    masked_corr = corr.mask(mask).iloc[::-1]

    text_annotations = masked_corr.round(2).fillna("").astype(str)

    # Create heatmap with annotations
    fig = go.Figure(
        data=go.Heatmap(
            z=masked_corr.values,
            x=corr.columns,
            y=corr.columns[::-1],
            colorscale="rdbu_r",
            zmin=-1,
            zmax=1,
            text=text_annotations,  # Add annotations
            texttemplate="%{text}",
            textfont=dict(size=13),
            hoverinfo="z",  # Display annotations on hover
        )
    )

    # Update layout for better aesthetics
    _ = fig.update_layout(  ## pyright: ignore[reportUnknownMemberType]
        title="Feature Correlation Matrix",
        xaxis=dict(title="Features", tickangle=45, automargin=True),
        yaxis=dict(title="Features", automargin=True),
        autosize=False,
        width=plot_size[0],
        height=plot_size[1],
        coloraxis_colorbar=dict(title="Correlation", tickvals=[-1, 0, 1]),
    )

    return fig


def plot_box_diagram(
    df: pd.DataFrame, x_col: str, y_col: str, plot_size: tuple[int, int] = (800, 600)
):
    # Создаем фигуру
    fig = go.Figure()

    # Добавляем коробчатую диаграмму
    _ = fig.add_trace(  ## pyright: ignore[reportUnknownMemberType]
        go.Box(
            x=df[x_col],
            y=df[y_col],
            name="Box Plot",
            # marker_color="lightgray",
            boxpoints="all",
            boxmean=True,  # Показывает среднее значение
        )
    )

    # Обновляем макет
    _ = fig.update_layout(  ## pyright: ignore[reportUnknownMemberType]
        title=f"Box Plot of {y_col} on {x_col}",
        xaxis=dict(
            title=x_col,
            # tickmode="array",
            # tickvals=[0, 1],
            # ticktext=["Weekday", "Weekend"],
        ),
        yaxis_title=y_col,
        template="plotly_white",
        width=plot_size[0],
        height=plot_size[1],
        showlegend=True,
        legend=dict(
            title="Legend",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig


def plot_density_diagram(
    df: pd.DataFrame,
    numeric_col: str,
    categorical_col: str,
    labels_map: dict[Any, str] | None = None,
    plot_size=(800, 600),
):
    fig = go.Figure()

    color_list = colors.qualitative.Plotly

    categorical_vals = sorted(df[categorical_col].unique())

    if not labels_map:
        labels_map = {}
        for cat in categorical_vals:
            str_cat = str(cat)
            labels_map.update({cat: str_cat})

    for i, category in enumerate(categorical_vals):
        # Отфильтровываем данные для текущей группы
        group_data = df[df[categorical_col] == category][numeric_col]

        if (
            len(group_data) > 1
        ):  # Проверяем, есть ли достаточно данных для оценки плотности
            # Оценка плотности с использованием KDE
            kde = stats.gaussian_kde(group_data)
            x_range = np.linspace(group_data.min(), group_data.max(), 100)
            y_values = kde(x_range)

            # Добавляем график плотности
            _ = fig.add_trace(
                go.Scatter(  ## pyright: ignore[reportUnknownMemberType]
                    x=x_range,
                    y=y_values,
                    mode="lines",
                    name=labels_map[category],
                    line=dict(color=color_list[i % len(color_list)], width=2),
                    fill="tozeroy",
                    opacity=0.5,
                )
            )

    # Обновляем макет
    _ = fig.update_layout(  ## pyright: ignore[reportUnknownMemberType]
        title=f"Density Plot of {numeric_col} by {categorical_col} categories",
        xaxis_title=numeric_col,
        yaxis_title="Density",
        template="plotly_white",
        width=plot_size[0],
        height=plot_size[1],
        legend=dict(
            title="Legend",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
    )

    return fig
