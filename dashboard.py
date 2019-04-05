from nx_rf import AgentSkeleton, GetBasedonDegree_V1, GetBasedonDegree_V2, GetBasedonDegree_V3, RegressorAgent, selectiveWorld
import config as cfg


agents = []
agents += [AgentSkeleton() for i in range(2)]
agents += [GetBasedonDegree_V1() for i in range(2)]
agents += [GetBasedonDegree_V2() for i in range(2)]
agents += [GetBasedonDegree_V3() for i in range(2)]
test_world = selectiveWorld(agents, agent_power=3, min_degs=1, selection_proba = 0.2)
test_world.iterate(10, draw=False)

import dash
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([html.Div([dcc.Dropdown(id='agent_selector',
                                     options=[{'label': i.__name__, 'value': i.__name__} for i in cfg.valid_agents],
                                     placeholder='Select agents to add')], className='col no-gutters'),
              html.Div([dcc.Input(id='num_selector',
                                  inputmode='number',
                                  placeholder='Number of agents')], className='col-s'),

              html.Div([html.Button(children='Add', id='add_agents', className='btn no-gutters')], className='col'),
              html.Div([html.Button(children='Start', id='start', className='btn no-gutters')], className='col'),],
             className='row no-gutters'),

    dcc.Dropdown(id='agents', multi=True, value=[], options=[]),
    dcc.Slider(id='step-chooser',
               min=0,
               max=len(test_world.history)-1,
               marks={i: 'Step {}'.format(i) for i in range(len(test_world.history))},
               value=len(test_world.history)-1),
    
    cyto.Cytoscape(
        id='cytoscape',
        layout={'name': 'circle'},
        style={'width': '100%', 'height': '600px'},
        elements=[]
    )
])


@app.callback(output=[dash.dependencies.Output('agents', 'value'),
                      dash.dependencies.Output('agents', 'options')],
              inputs=[dash.dependencies.Input('add_agents', 'n_clicks_timestamp')],
              state=[dash.dependencies.State('agent_selector', 'value'),
                     dash.dependencies.State('num_selector', 'value')])
def refresh_agent(click, selected_agent, num_agents):
    if selected_agent is None:
        print('No agent seleced, add agent added')
    try:
        num_agents = int(num_agents)
    except:
        print(num_agents, 'is not a number')



@app.callback(dash.dependencies.Output('cytoscape', 'elements'),
              [dash.dependencies.Input('step-chooser', 'value')],
              state=[dash.dependencies.State('agents', 'value')])
def refresh_edges(step, vals):
    print(vals)
    nodes = [{'data': {'id':i.name, 'label': i.name}} for i in test_world.history[step].nodes]
    edges = [{'data': {'source': i[0].name, 'target': i[1].name}, 'selectable': False} for i in test_world.history[step].edges]
    
    return nodes+edges


app.css.append_css({'external_url': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css'})

if __name__ == '__main__':
    app.run_server(debug=False)
