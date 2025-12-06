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
    title="HAVOC - Red Team Toolkit",
    suppress_callback_exceptions=True
)

# Dark aggressive theme
COLORS = {
    'bg': '#0a0a0f',
    'card': '#12121a',
    'accent': '#ff0040',
    'accent2': '#00ffff',
    'text': '#ffffff',
    'text_dim': '#888888',
    'success': '#00ff00',
    'danger': '#ff0040',
    'warning': '#ffaa00',
    'rejected': '#666666',
}

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


def load_past_results(results_dir: str = "results") -> list:
    """Load past attack results"""
    results = []
    results_path = Path(results_dir)
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
        html.H1("HAVOC", style={
            'color': COLORS['accent'],
            'fontSize': '48px',
            'fontWeight': 'bold',
            'margin': '0',
            'textShadow': f'0 0 20px {COLORS["accent"]}'
        }),
        html.P("NIA Red Team Toolkit", style={
            'color': COLORS['text_dim'],
            'margin': '5px 0 0 0',
            'fontSize': '14px',
            'letterSpacing': '3px'
        }),
    ], style={'textAlign': 'center', 'padding': '30px 0'}),

    # API Configuration Section
    html.Div([
        html.Div([
            html.Label("OPENROUTER API KEY", style={
                'color': COLORS['accent2'],
                'fontWeight': 'bold',
                'fontSize': '11px',
                'letterSpacing': '2px',
                'marginBottom': '8px',
                'display': 'block'
            }),
            dcc.Input(
                id='api-key-input',
                type='password',
                placeholder='sk-or-v1-...',
                value=os.environ.get('OPENROUTER_API_KEY', ''),
                style={
                    'width': '100%',
                    'backgroundColor': COLORS['card'],
                    'color': COLORS['text'],
                    'border': f'1px solid {COLORS["accent2"]}50',
                    'borderRadius': '5px',
                    'padding': '12px',
                    'fontSize': '14px',
                }
            ),
        ], style={'flex': '2', 'marginRight': '15px'}),

        html.Div([
            html.Label("TARGET MODEL", style={
                'color': COLORS['accent2'],
                'fontWeight': 'bold',
                'fontSize': '11px',
                'letterSpacing': '2px',
                'marginBottom': '8px',
                'display': 'block'
            }),
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
                style={'backgroundColor': COLORS['card']},
                className='dark-dropdown'
            ),
            dcc.Input(
                id='custom-model-input',
                type='text',
                placeholder='e.g., mistralai/mistral-large',
                style={
                    'width': '100%',
                    'backgroundColor': COLORS['card'],
                    'color': COLORS['text'],
                    'border': f'1px solid {COLORS["accent2"]}30',
                    'borderRadius': '5px',
                    'padding': '8px',
                    'fontSize': '12px',
                    'marginTop': '5px',
                    'display': 'none'
                }
            ),
        ], style={'flex': '1', 'marginRight': '15px', 'minWidth': '200px'}),

        html.Div([
            html.Label("JUDGE MODEL", style={
                'color': COLORS['accent2'],
                'fontWeight': 'bold',
                'fontSize': '11px',
                'letterSpacing': '2px',
                'marginBottom': '8px',
                'display': 'block'
            }),
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
                style={'backgroundColor': COLORS['card']},
                className='dark-dropdown'
            ),
        ], style={'flex': '1', 'minWidth': '200px'}),
    ], style={
        'display': 'flex',
        'padding': '15px 40px',
        'backgroundColor': COLORS['card'],
        'margin': '0 20px 15px 20px',
        'borderRadius': '10px',
        'border': f'1px solid {COLORS["accent2"]}30',
        'flexWrap': 'wrap',
        'gap': '10px'
    }),

    # Attack Input Section
    html.Div([
        html.Div([
            html.Label("TARGET OBJECTIVE", style={
                'color': COLORS['accent'],
                'fontWeight': 'bold',
                'fontSize': '12px',
                'letterSpacing': '2px',
                'marginBottom': '10px',
                'display': 'block'
            }),
            dcc.Textarea(
                id='attack-input',
                placeholder='Enter your attack objective here...\n\nExample: "How to pick a lock"',
                style={
                    'width': '100%',
                    'height': '100px',
                    'backgroundColor': COLORS['card'],
                    'color': COLORS['text'],
                    'border': f'1px solid {COLORS["accent"]}',
                    'borderRadius': '5px',
                    'padding': '15px',
                    'fontSize': '16px',
                    'resize': 'none'
                }
            ),
        ], style={'flex': '2', 'marginRight': '20px'}),

        html.Div([
            html.Label("TEMPLATES", style={
                'color': COLORS['accent'],
                'fontWeight': 'bold',
                'fontSize': '12px',
                'letterSpacing': '2px',
                'marginBottom': '10px',
                'display': 'block'
            }),
            html.Div([
                html.Span(id='template-count', children="0 templates loaded", style={
                    'color': COLORS['text_dim'],
                    'fontSize': '14px',
                }),
            ], style={'marginBottom': '15px'}),
            html.Button(
                "LAUNCH HAVOC",
                id='launch-btn',
                style={
                    'width': '100%',
                    'padding': '15px',
                    'backgroundColor': COLORS['accent'],
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '5px',
                    'fontSize': '16px',
                    'fontWeight': 'bold',
                    'cursor': 'pointer',
                    'letterSpacing': '2px',
                    'boxShadow': f'0 0 20px {COLORS["accent"]}50'
                }
            ),
        ], style={'flex': '1', 'minWidth': '250px'}),
    ], style={
        'display': 'flex',
        'padding': '20px 40px',
        'backgroundColor': COLORS['card'],
        'margin': '0 20px',
        'borderRadius': '10px',
        'border': f'1px solid {COLORS["accent"]}30'
    }),

    # Status & Progress
    html.Div([
        html.Div(id='attack-status', style={
            'color': COLORS['text_dim'],
            'textAlign': 'center',
            'padding': '15px',
            'fontSize': '14px'
        }),
        dcc.Loading(
            id="loading",
            type="dot",
            color=COLORS['accent'],
            children=html.Div(id='loading-output')
        ),
    ]),

    # Stats Cards
    html.Div(id='stats-cards', style={
        'display': 'flex',
        'justifyContent': 'center',
        'flexWrap': 'wrap',
        'padding': '20px'
    }),

    # Charts
    html.Div([
        html.Div([
            dcc.Graph(id='results-pie')
        ], style={
            'flex': '1',
            'minWidth': '350px',
            'backgroundColor': COLORS['card'],
            'borderRadius': '10px',
            'margin': '10px',
            'padding': '10px'
        }),
        html.Div([
            dcc.Graph(id='template-bar')
        ], style={
            'flex': '2',
            'minWidth': '500px',
            'backgroundColor': COLORS['card'],
            'borderRadius': '10px',
            'margin': '10px',
            'padding': '10px'
        }),
    ], style={'display': 'flex', 'flexWrap': 'wrap', 'padding': '0 20px'}),

    # Results Table
    html.Div([
        html.H3("ATTACK RESULTS", style={
            'color': COLORS['accent'],
            'letterSpacing': '2px',
            'marginBottom': '15px'
        }),
        html.Div(id='results-table')
    ], style={
        'backgroundColor': COLORS['card'],
        'borderRadius': '10px',
        'margin': '20px',
        'padding': '20px'
    }),

    # Past Runs Section
    html.Div([
        html.H3("PAST RUNS", style={
            'color': COLORS['accent2'],
            'letterSpacing': '2px',
            'marginBottom': '15px'
        }),
        dcc.Dropdown(
            id='past-run-selector',
            placeholder="Select a past run to view...",
            style={'backgroundColor': COLORS['card'], 'marginBottom': '15px'}
        ),
        html.Div(id='past-run-display')
    ], style={
        'backgroundColor': COLORS['card'],
        'borderRadius': '10px',
        'margin': '20px',
        'padding': '20px'
    }),

    # Hidden stores
    dcc.Store(id='results-store'),
    dcc.Interval(id='refresh-interval', interval=5000, disabled=True),

], style={
    'backgroundColor': COLORS['bg'],
    'minHeight': '100vh',
    'fontFamily': "'Segoe UI', Arial, sans-serif"
})


def create_stat_card(title, value, color):
    """Create a stat card"""
    return html.Div([
        html.Div(title, style={
            'color': COLORS['text_dim'],
            'fontSize': '11px',
            'letterSpacing': '1px',
            'marginBottom': '5px'
        }),
        html.Div(str(value), style={
            'color': color,
            'fontSize': '32px',
            'fontWeight': 'bold'
        })
    ], style={
        'backgroundColor': COLORS['card'],
        'padding': '20px 30px',
        'borderRadius': '10px',
        'margin': '5px',
        'textAlign': 'center',
        'border': f'1px solid {color}30'
    })


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
    prevent_initial_call=False
)
def toggle_custom_model_input(model_value):
    """Show/hide custom model input based on selection"""
    base_style = {
        'width': '100%',
        'backgroundColor': COLORS['card'],
        'color': COLORS['text'],
        'border': f'1px solid {COLORS["accent2"]}30',
        'borderRadius': '5px',
        'padding': '8px',
        'fontSize': '12px',
        'marginTop': '5px',
    }
    if model_value == 'custom':
        base_style['display'] = 'block'
    else:
        base_style['display'] = 'none'
    return base_style


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
     State('api-key-input', 'value')],
    prevent_initial_call=True
)
def launch_attack(n_clicks, objective, model, custom_model, judge_model, api_key):
    """Launch attack with all templates"""

    # Empty figures
    empty_fig = go.Figure()
    empty_fig.update_layout(
        paper_bgcolor=COLORS['card'],
        plot_bgcolor=COLORS['card'],
        font_color=COLORS['text']
    )

    if not objective or not objective.strip():
        return (
            [],
            empty_fig,
            empty_fig,
            html.P("Enter an objective to attack", style={'color': COLORS['text_dim']}),
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
            html.P("Enter your OpenRouter API key!", style={'color': COLORS['danger']}),
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
                html.P("Please enter a custom model name!", style={'color': COLORS['danger']}),
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

        runner = BatchRunner(
            template_engine=engine,
            target=target,
            scorer=scorer,
            concurrency=10,
            output_dir="results"
        )

        # Run attack
        result = runner.run_batch(objectives=[objective.strip()])

        # Save results
        filepath = result.save("results")

        stats = result.statistics

        # Build stats cards
        cards = [
            create_stat_card("TOTAL", stats.get('total', 0), COLORS['text']),
            create_stat_card("SUCCESS", stats.get('harmful', 0), COLORS['danger']),
            create_stat_card("REJECTED", stats.get('rejected', 0), COLORS['rejected']),
            create_stat_card("SAFE", stats.get('safe', 0), COLORS['success']),
            create_stat_card("SUCCESS %", f"{stats.get('attack_success_rate', 0)*100:.1f}%", COLORS['accent']),
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
            marker_colors=[COLORS['rejected'], COLORS['danger'], COLORS['success'], COLORS['warning']],
            hole=0.4
        )])
        pie_fig.update_layout(
            title={'text': 'Result Distribution', 'font': {'color': COLORS['text']}},
            paper_bgcolor=COLORS['card'],
            plot_bgcolor=COLORS['card'],
            font_color=COLORS['text'],
            showlegend=True,
            legend=dict(font=dict(color=COLORS['text']))
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
                marker_color=COLORS['accent']
            )])
            bar_fig.update_layout(
                title={'text': 'Successful Attack Templates', 'font': {'color': COLORS['text']}},
                paper_bgcolor=COLORS['card'],
                plot_bgcolor=COLORS['card'],
                font_color=COLORS['text'],
                xaxis_tickangle=-45,
                yaxis_title='Success Rate %'
            )
        else:
            bar_fig = empty_fig
            bar_fig.update_layout(
                title={'text': 'No Successful Attacks', 'font': {'color': COLORS['text']}}
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
            style_table={'overflowX': 'auto'},
            style_cell={
                'backgroundColor': COLORS['card'],
                'color': COLORS['text'],
                'textAlign': 'left',
                'padding': '10px',
                'maxWidth': '300px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis'
            },
            style_header={
                'backgroundColor': COLORS['bg'],
                'fontWeight': 'bold',
                'color': COLORS['accent']
            },
            style_data_conditional=[
                {'if': {'filter_query': '{Result} = HARMFUL'}, 'backgroundColor': f'{COLORS["danger"]}30'},
                {'if': {'filter_query': '{Result} = REJECTED'}, 'backgroundColor': f'{COLORS["rejected"]}30'},
                {'if': {'filter_query': '{Result} = SAFE'}, 'backgroundColor': f'{COLORS["success"]}20'},
            ],
            page_size=15,
            filter_action='native',
            sort_action='native',
        )

        judge_info = f" | Judge: {judge_model.split('/')[-1]}" if use_judge else " | Rule-based scoring only"
        status = f"Attack complete! {stats.get('total', 0)} templates tested against {target_model.split('/')[-1]}{judge_info}. Results saved."

        return cards, pie_fig, bar_fig, table, status, "", result.to_dict()

    except Exception as e:
        return (
            [],
            empty_fig,
            empty_fig,
            html.P(f"Error: {str(e)}", style={'color': COLORS['danger']}),
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
    """Display a past run's results"""
    if not run_id:
        return html.P("Select a run to view", style={'color': COLORS['text_dim']})

    results = load_past_results()
    batch = next((r for r in results if r.get('run_id') == run_id), None)

    if not batch:
        return html.P("Run not found", style={'color': COLORS['text_dim']})

    stats = batch.get('statistics', {})

    return html.Div([
        html.Div([
            html.Span(f"Model: {batch.get('target_model', 'Unknown')}", style={'marginRight': '20px'}),
            html.Span(f"Total: {stats.get('total', 0)}", style={'marginRight': '20px'}),
            html.Span(f"Success: {stats.get('harmful', 0)} ({stats.get('attack_success_rate', 0)*100:.1f}%)",
                     style={'color': COLORS['danger'], 'marginRight': '20px'}),
            html.Span(f"Rejected: {stats.get('rejected', 0)}", style={'color': COLORS['rejected']}),
        ], style={'color': COLORS['text'], 'marginBottom': '15px'}),

        html.Details([
            html.Summary("View Objectives", style={'color': COLORS['accent2'], 'cursor': 'pointer'}),
            html.Pre(
                json.dumps(batch.get('objectives', []), indent=2),
                style={'color': COLORS['text_dim'], 'fontSize': '12px'}
            )
        ])
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

    NIA Red Team Toolkit
    Dashboard: http://{host}:{port}

    Press Ctrl+C to stop
    """)
    app.run_server(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run_dashboard()
