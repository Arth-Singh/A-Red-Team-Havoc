#!/usr/bin/env python3
"""
HAVOC Dashboard
NIA Red Team Toolkit - Interactive Attack Interface
"""

import json
import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime
from threading import Thread

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.template_engine import TemplateEngine
from src.core.target import OpenRouterTarget
from src.core.scorer import Scorer, ScoreResult
from src.core.batch_runner import BatchRunner

# Initialize Dash app
app = dash.Dash(
    __name__,
    title="A.R.T.H - A Red Team Helper",
    suppress_callback_exceptions=True,
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap'
    ]
)

# Global state
ENGINE = None
CURRENT_RESULTS = None


def init_engine():
    """Initialize template engine"""
    global ENGINE
    if ENGINE is None:
        ENGINE = TemplateEngine('templates')
        ENGINE.load_all_templates()
    return ENGINE


def load_past_results() -> list:
    """Load past attack results"""
    results = []
    # Use absolute path relative to this file
    results_path = Path(__file__).parent.parent / "results"
    if results_path.exists():
        for f in results_path.glob("*.json"):
            try:
                with open(f) as file:
                    data = json.load(file)
                    data['_file'] = f.name
                    results.append(data)
            except:
                pass
    results.sort(key=lambda x: x.get('start_time', ''), reverse=True)
    return results


# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("A.R.T.H"),
        html.P("A Red Team Helper"),
    ], className='header'),

    # API Configuration Section
    html.Div([
        html.Div([
            html.Label("OPENROUTER API KEY"),
            dcc.Input(
                id='api-key-input',
                type='password',
                placeholder='sk-or-v1-...',
                value=os.environ.get('OPENROUTER_API_KEY', ''),
                className='dark-input'
            ),
        ], className='control-group'),

        html.Div([
            html.Label("TARGET MODEL"),
            dcc.Dropdown(
                id='model-selector',
                options=[
                    {'label': 'Llama 3.1 8B', 'value': 'meta-llama/llama-3.1-8b-instruct'},
                    {'label': 'Llama 3.1 70B', 'value': 'meta-llama/llama-3.1-70b-instruct'},
                    {'label': 'Llama 3.3 70B', 'value': 'meta-llama/llama-3.3-70b-instruct'},
                    {'label': 'Mistral 7B', 'value': 'mistralai/mistral-7b-instruct'},
                    {'label': 'Mixtral 8x7B', 'value': 'mistralai/mixtral-8x7b-instruct'},
                    {'label': 'Gemini Flash 1.5', 'value': 'google/gemini-flash-1.5'},
                    {'label': 'Gemini Pro 1.5', 'value': 'google/gemini-pro-1.5'},
                    {'label': 'GPT-4o', 'value': 'openai/gpt-4o'},
                    {'label': 'GPT-4o Mini', 'value': 'openai/gpt-4o-mini'},
                    {'label': 'GPT-4 Turbo', 'value': 'openai/gpt-4-turbo'},
                    {'label': 'Claude 3.5 Sonnet', 'value': 'anthropic/claude-3.5-sonnet'},
                    {'label': 'Claude 3 Haiku', 'value': 'anthropic/claude-3-haiku'},
                    {'label': 'Claude 3 Opus', 'value': 'anthropic/claude-3-opus'},
                    {'label': 'Qwen 2.5 72B', 'value': 'qwen/qwen-2.5-72b-instruct'},
                    {'label': 'DeepSeek Chat', 'value': 'deepseek/deepseek-chat'},
                    {'label': 'DeepSeek R1', 'value': 'deepseek/deepseek-r1'},
                    {'label': 'Custom (enter below)', 'value': 'custom'},
                ],
                value=os.environ.get('TARGET_MODEL', 'meta-llama/llama-3.1-8b-instruct'),
                className='dark-dropdown'
            ),
            dcc.Input(
                id='custom-model-input',
                type='text',
                placeholder='e.g., mistralai/mistral-large',
                className='dark-input',
                style={'display': 'none'}
            ),
        ], className='control-group'),

        html.Div([
            html.Label("JUDGE MODEL"),
            dcc.Dropdown(
                id='judge-model-selector',
                options=[
                    {'label': 'GPT-4o (Recommended)', 'value': 'openai/gpt-4o'},
                    {'label': 'GPT-4o Mini (Faster)', 'value': 'openai/gpt-4o-mini'},
                    {'label': 'GPT-4 Turbo', 'value': 'openai/gpt-4-turbo'},
                    {'label': 'Claude 3.5 Sonnet', 'value': 'anthropic/claude-3.5-sonnet'},
                    {'label': 'Claude 3 Haiku (Faster)', 'value': 'anthropic/claude-3-haiku'},
                    {'label': 'Gemini Pro 1.5', 'value': 'google/gemini-pro-1.5'},
                    {'label': 'Gemini Flash 1.5', 'value': 'google/gemini-flash-1.5'},
                    {'label': 'None (Rule-based only)', 'value': 'none'},
                ],
                value=os.environ.get('JUDGE_MODEL', 'openai/gpt-4o'),
                className='dark-dropdown'
            ),
        ], className='control-group'),

        html.Div([
            html.Label("ITERATIONS PER TEMPLATE"),
            dcc.Dropdown(
                id='iterations-selector',
                options=[
                    {'label': '1x (Fast)', 'value': 1},
                    {'label': '3x', 'value': 3},
                    {'label': '5x (Recommended)', 'value': 5},
                    {'label': '10x (Thorough)', 'value': 10},
                ],
                value=5,
                className='dark-dropdown'
            ),
        ], className='control-group'),
    ], className='control-panel'),

    # Attack Input Section
    html.Div([
        html.Div([
            html.Label("TARGET OBJECTIVE"),
            dcc.Textarea(
                id='attack-input',
                placeholder='Enter your attack objective here...\n\nExample: "How to pick a lock"',
                className='attack-textarea'
            ),
        ], className='attack-input-group'),

        html.Div([
            html.Div(id='template-count', className='template-count'),
            html.Button("LAUNCH HAVOC", id='launch-btn', className='launch-btn'),
        ], className='launch-button-group'),
    ], className='attack-panel'),

    # Status & Progress
    html.Div([
        html.Div(id='attack-status'),
        dcc.Loading(
            id="loading",
            type="dot",
            color="#ff0040",
            children=html.Div(id='loading-output')
        ),
    ], className='status-section'),

    # Stats Cards
    html.Div(id='stats-cards', className='stats-container'),

    # Charts
    html.Div([
        html.Div(dcc.Graph(id='results-pie'), className='chart-card'),
        html.Div(dcc.Graph(id='template-bar'), className='chart-card'),
    ], className='charts-container'),

    # Results Table
    html.Div([
        html.H3("ATTACK RESULTS", className='section-title'),
        html.Div(id='results-table', className='results-table')
    ], className='results-section'),

    # Past Runs Section
    html.Div([
        html.H3("PAST RUNS", className='section-title'),
        dcc.Dropdown(
            id='past-run-selector',
            placeholder="Select a past run to view...",
            className='dark-dropdown'
        ),
        html.Div(id='past-run-display')
    ], className='past-runs-section'),

    # Hidden stores
    dcc.Store(id='results-store'),
    dcc.Interval(id='refresh-interval', interval=5000, disabled=True),

])


def create_stat_card(title, value, color_class):
    """Create a stat card"""
    return html.Div([
        html.Div(title, className='stat-card-title'),
        html.Div(str(value), className=f'stat-card-value {color_class}')
    ], className='stat-card')


@app.callback(
    Output('past-run-selector', 'options'),
    Input('results-store', 'data'),
    prevent_initial_call=False
)
def update_past_runs(_):
    """Update past runs dropdown"""
    results = load_past_results()
    return [
        {
            'label': f"{r.get('run_id', '?')} | {r.get('target_model', '').split('/')[-1]} | {r.get('total_attacks', 0)} attacks | {r.get('statistics', {}).get('attack_success_rate', 0)*100:.1f}% success",
            'value': r.get('run_id')
        }
        for r in results
    ]


@app.callback(
    Output('template-count', 'children'),
    Input('launch-btn', 'n_clicks'),
    prevent_initial_call=False
)
def update_template_count(_):
    """Update template count on load"""
    engine = init_engine()
    count = len(engine.get_all_templates())
    return f"{count} attack templates loaded"


@app.callback(
    Output('custom-model-input', 'style'),
    Input('model-selector', 'value'),
    State('custom-model-input', 'style'),
    prevent_initial_call=False
)
def toggle_custom_model_input(model_value, current_style):
    """Show/hide custom model input based on selection"""
    if model_value == 'custom':
        current_style['display'] = 'block'
    else:
        current_style['display'] = 'none'
    return current_style


@app.callback(
    [Output('stats-cards', 'children'),
     Output('results-pie', 'figure'),
     Output('template-bar', 'figure'),
     Output('results-table', 'children'),
     Output('attack-status', 'children'),
     Output('loading-output', 'children'),
     Output('results-store', 'data')],
    [Input('launch-btn', 'n_clicks')],
    [State('attack-input', 'value'),
     State('model-selector', 'value'),
     State('custom-model-input', 'value'),
     State('judge-model-selector', 'value'),
     State('api-key-input', 'value'),
     State('iterations-selector', 'value')],
    prevent_initial_call=True
)
def launch_attack(n_clicks, objective, model, custom_model, judge_model, api_key, iterations):
    """Launch attack with all templates"""

    # Empty figures
    empty_fig = go.Figure()
    empty_fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff'
    )

    if not objective or not objective.strip():
        return (
            [],
            empty_fig,
            empty_fig,
            html.P("Enter an objective to attack"),
            "Ready to launch",
            "",
            None
        )

    # Check API key - use from input or fall back to env
    if not api_key:
        api_key = os.environ.get('OPENROUTER_API_KEY')

    if not api_key:
        return (
            [],
            empty_fig,
            empty_fig,
            html.P("Enter your OpenRouter API key!"),
            "ERROR: OpenRouter API key required",
            "",
            None
        )

    try:
        # Initialize
        engine = init_engine()

        # Use custom model if selected
        target_model = custom_model if model == 'custom' and custom_model else model
        if not target_model or target_model == 'custom':
            return (
                [],
                empty_fig,
                empty_fig,
                html.P("Please enter a custom model name!"),
                "ERROR: Custom model name required",
                "",
                None
            )

        target = OpenRouterTarget(api_key=api_key, model=target_model)

        # Setup LLM judge for scoring non-rejected responses
        use_judge = judge_model and judge_model != 'none'
        if use_judge:
            judge_target = OpenRouterTarget(api_key=api_key, model=judge_model)
            scorer = Scorer(judge_target=judge_target, use_judge=True)
        else:
            scorer = Scorer(use_judge=False)  # Rule-based only

        # Use absolute path for results
        results_dir = Path(__file__).parent.parent / "results"
        results_dir.mkdir(exist_ok=True)

        runner = BatchRunner(
            template_engine=engine,
            target=target,
            scorer=scorer,
            concurrency=10,
            output_dir=str(results_dir)
        )

        # Default iterations if not set
        if not iterations:
            iterations = 5

        print(f"\n[HAVOC] Starting attack against {target_model}")
        print(f"[HAVOC] Objective: {objective.strip()[:100]}...")
        print(f"[HAVOC] Templates: {len(engine.get_all_templates())}")
        print(f"[HAVOC] Iterations per template: {iterations}")
        print(f"[HAVOC] Judge: {judge_model if use_judge else 'Rule-based only'}")

        # Run attack with specified iterations
        result = runner.run_batch(objectives=[objective.strip()], iterations=iterations)

        print(f"[HAVOC] Attack complete! Saving results...")

        # Save results
        filepath = result.save(str(results_dir))
        print(f"[HAVOC] Results saved to: {filepath}")

        stats = result.statistics

        # Build stats cards
        cards = [
            create_stat_card("TOTAL", stats.get('total', 0), 'text-white'),
            create_stat_card("SUCCESS", stats.get('harmful', 0), 'text-danger'),
            create_stat_card("REJECTED", stats.get('rejected', 0), 'text-rejected'),
            create_stat_card("SAFE", stats.get('safe', 0), 'text-success'),
            create_stat_card("SUCCESS %", f"{stats.get('attack_success_rate', 0)*100:.1f}%", 'text-accent'),
        ]

        # Pie chart
        pie_fig = go.Figure(data=[go.Pie(
            labels=['Rejected', 'Harmful', 'Safe', 'Error'],
            values=[
                stats.get('rejected', 0),
                stats.get('harmful', 0),
                stats.get('safe', 0),
                stats.get('error', 0) + stats.get('undetermined', 0)
            ],
            marker_colors=['#666666', '#ff0040', '#00ff00', '#ffaa00'],
            hole=0.5,
            textfont_size=14
        )])
        pie_fig.update_layout(
            title={'text': 'Result Distribution', 'font': {'color': '#ffffff'}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            showlegend=True,
            legend=dict(font=dict(color='#ffffff'))
        )

        # Bar chart - top successful templates
        template_stats = stats.get('per_template', {})
        successful = [(name, data.get('success_rate', 0))
                      for name, data in template_stats.items()
                      if data.get('success_rate', 0) > 0]
        successful.sort(key=lambda x: x[1], reverse=True)
        top_20 = successful[:20]

        if top_20:
            bar_fig = go.Figure(data=[go.Bar(
                x=[t[0][:25] for t in top_20],
                y=[t[1] * 100 for t in top_20],
                marker_color='#ff0040'
            )])
            bar_fig.update_layout(
                title={'text': 'Top 20 Successful Attack Templates', 'font': {'color': '#ffffff'}},
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#ffffff',
                xaxis_tickangle=-45,
                yaxis_title='Success Rate %'
            )
        else:
            bar_fig = empty_fig
            bar_fig.update_layout(
                title={'text': 'No Successful Attacks', 'font': {'color': '#ffffff'}}
            )

        # Results table
        rows = []
        for r in result.results:
            rows.append({
                'Template': r.template_name,
                'Result': r.score.result.value.upper(),
                'Latency': f"{r.response.latency_ms:.0f}ms",
                'Response': r.response.response_text[:200] + '...' if len(r.response.response_text) > 200 else r.response.response_text
            })

        df = pd.DataFrame(rows)

        table = dash_table.DataTable(
            data=df.to_dict('records'),
            columns=[{'name': c, 'id': c} for c in df.columns],
            page_size=15,
            filter_action='native',
            sort_action='native',
            style_data_conditional=[
                {'if': {'filter_query': '{Result} = HARMFUL'}, 'backgroundColor': '#ff004030'},
                {'if': {'filter_query': '{Result} = REJECTED'}, 'backgroundColor': '#66666630'},
                {'if': {'filter_query': '{Result} = SAFE'}, 'backgroundColor': '#00ff0020'},
            ],
        )

        judge_info = f" | Judge: {judge_model.split('/')[-1]}" if use_judge else " | Rule-based scoring only"
        num_templates = len(engine.get_all_templates())
        status = f"Attack complete! {num_templates} templates × {iterations} iterations = {stats.get('total', 0)} attacks against {target_model.split('/')[-1]}{judge_info}. Results saved."

        return cards, pie_fig, bar_fig, table, status, "", result.to_dict()

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"\n[HAVOC ERROR] {str(e)}")
        print(error_trace)
        return (
            [],
            empty_fig,
            empty_fig,
            html.P(f"Error: {str(e)}"),
            f"ERROR: {str(e)}",
            "",
            None
        )


@app.callback(
    Output('past-run-display', 'children'),
    Input('past-run-selector', 'value'),
    prevent_initial_call=True
)
def display_past_run(run_id):
    """Display a past run's full results with charts and table"""
    if not run_id:
        return html.P("Select a run to view")

    results = load_past_results()
    batch = next((r for r in results if r.get('run_id') == run_id), None)

    if not batch:
        return html.P("Run not found")

    stats = batch.get('statistics', {})
    results_data = batch.get('results', [])

    # Build stat cards
    stat_cards = html.Div([
        html.Div([
            html.Div("TOTAL", className='stat-card-title'),
            html.Div(str(stats.get('total', 0)), className='stat-card-value text-white')
        ], className='stat-card'),
        html.Div([
            html.Div("SUCCESS", className='stat-card-title'),
            html.Div(str(stats.get('harmful', 0)), className='stat-card-value text-danger')
        ], className='stat-card'),
        html.Div([
            html.Div("REJECTED", className='stat-card-title'),
            html.Div(str(stats.get('rejected', 0)), className='stat-card-value text-rejected')
        ], className='stat-card'),
        html.Div([
            html.Div("SAFE", className='stat-card-title'),
            html.Div(str(stats.get('safe', 0)), className='stat-card-value text-success')
        ], className='stat-card'),
        html.Div([
            html.Div("SUCCESS %", className='stat-card-title'),
            html.Div(f"{stats.get('attack_success_rate', 0)*100:.1f}%", className='stat-card-value text-accent')
        ], className='stat-card'),
    ], className='stats-container')

    # Pie chart
    pie_fig = go.Figure(data=[go.Pie(
        labels=['Rejected', 'Harmful', 'Safe', 'Error'],
        values=[
            stats.get('rejected', 0),
            stats.get('harmful', 0),
            stats.get('safe', 0),
            stats.get('error', 0) + stats.get('undetermined', 0)
        ],
        marker_colors=['#666666', '#ff0040', '#00ff00', '#ffaa00'],
        hole=0.5,
        textfont_size=14
    )])
    pie_fig.update_layout(
        title={'text': 'Result Distribution', 'font': {'color': '#ffffff'}},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        showlegend=True,
        legend=dict(font=dict(color='#ffffff')),
        height=350
    )

    # Bar chart - top successful templates
    template_stats = stats.get('per_template', {})
    successful = [(name, data.get('success_rate', 0))
                  for name, data in template_stats.items()
                  if data.get('success_rate', 0) > 0]
    successful.sort(key=lambda x: x[1], reverse=True)
    top_20 = successful[:20]

    if top_20:
        bar_fig = go.Figure(data=[go.Bar(
            x=[t[0][:25] for t in top_20],
            y=[t[1] * 100 for t in top_20],
            marker_color='#ff0040'
        )])
        bar_fig.update_layout(
            title={'text': 'Top Successful Attack Templates', 'font': {'color': '#ffffff'}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            xaxis_tickangle=-45,
            yaxis_title='Success Rate %',
            height=350
        )
    else:
        bar_fig = go.Figure()
        bar_fig.update_layout(
            title={'text': 'No Successful Attacks', 'font': {'color': '#ffffff'}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            height=350
        )

    # Results table
    rows = []
    for r in results_data:
        response_text = r.get('response', {}).get('response_text', '')
        rows.append({
            'Template': r.get('template_name', ''),
            'Result': r.get('score', {}).get('result', '').upper(),
            'Response': response_text[:200] + '...' if len(response_text) > 200 else response_text
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=['Template', 'Result', 'Response'])

    table = dash_table.DataTable(
        data=df.to_dict('records'),
        columns=[{'name': c, 'id': c} for c in df.columns],
        page_size=15,
        filter_action='native',
        sort_action='native',
        style_data_conditional=[
            {'if': {'filter_query': '{Result} = HARMFUL'}, 'backgroundColor': '#ff004030'},
            {'if': {'filter_query': '{Result} = REJECTED'}, 'backgroundColor': '#66666630'},
            {'if': {'filter_query': '{Result} = SAFE'}, 'backgroundColor': '#00ff0020'},
        ],
    )

    return html.Div([
        # Header info
        html.Div([
            html.H4(f"Run: {run_id}", style={'color': '#ff0040', 'margin': '0'}),
            html.P(f"Model: {batch.get('target_model', 'Unknown')} | Time: {batch.get('start_time', '')[:19]}",
                   style={'color': '#888', 'margin': '5px 0'}),
            html.P(f"Objectives: {', '.join(batch.get('objectives', []))[:100]}...",
                   style={'color': '#aaa', 'margin': '5px 0', 'fontSize': '12px'}),
        ], style={'marginBottom': '20px'}),

        # Stats
        stat_cards,

        # Charts side by side
        html.Div([
            html.Div([dcc.Graph(figure=pie_fig)], className='chart-card', style={'flex': '1', 'minWidth': '300px'}),
            html.Div([dcc.Graph(figure=bar_fig)], className='chart-card', style={'flex': '2', 'minWidth': '400px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '15px', 'marginBottom': '20px'}),

        # Results table
        html.Div([
            html.H4("Attack Results", style={'color': '#00ffff', 'marginBottom': '10px'}),
            table
        ]),
    ])


def run_dashboard(host: str = "127.0.0.1", port: int = 8050, debug: bool = True):
    """Run HAVOC dashboard"""
    print(f"""
    ██╗  ██╗ █████╗ ██╗   ██╗ ██████╗  ██████╗
    ██║  ██║██╔══██╗██║   ██║██╔═══██╗██╔════╝
    ███████║███████║██║   ██║██║   ██║██║
    ██╔══██║██╔══██║╚██╗ ██╔╝██║   ██║██║
    ██║  ██║██║  ██║ ╚████╔╝ ╚██████╔╝╚██████╗
    ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝   ╚═════╝  ╚═════╝

    A.R.T.H - A Red Team Helper
    Dashboard: http://{host}:{port}

    Press Ctrl+C to stop
    """)
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard()
