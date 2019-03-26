from sklearn.ensemble import RandomForestRegressor
from nx_rf import GetBasedonDegree_V3, RegressorAgent, selectiveWorld

agents = []
agents += [GetBasedonDegree_V3() for i in range(10)]
#agents += [RegressorAgent(regressor=RandomForestRegressor()) for i in range(25)]
test_world = selectiveWorld(agents, agent_power=5, min_degs=1, selection_proba = 0.3)
test_world.iterate(10, draw = False)

import dash
import dash_core_components as dcc
import dash_cytoscape as cyto
import dash_html_components as html

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Slider(id='step-chooser',
               min=0,
               max=len(test_world.history),
               marks={i: 'Step {}'.format(i) for i in range(len(test_world.history))},
               value=len(test_world.history)),
    
    cyto.Cytoscape(
        id='cytoscape',
        layout={'name': 'circle'},
        style={'width': '100%', 'height': '600px'},
        elements=[]
    )
])

@app.callback(dash.dependencies.Output('cytoscape', 'elements'),
              [dash.dependencies.Input('step-chooser', 'value')])
def refresh_edges(step):
    nodes = [{'data': {'id':i.name, 'label': i.name}} for i in test_world.history[step].nodes]
    edges = [{'data': {'source': i[0].name, 'target': i[1].name}} for i in test_world.history[step].edges]
    
    return nodes+edges


if __name__ == '__main__':
    app.run_server(debug=False)