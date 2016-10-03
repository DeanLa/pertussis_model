from pertussis import *
import os
from datetime import datetime as dt
import webbrowser

def html_table(dict_of_vars, persist=('a_l', 'a_u', 'f', 'epsilon_ap', 'epsilon_wp')):
    table = '''<table>
                    <tr><th>Variable</th><th>Value</th><th>Type</th>
                    {}
                </table>'''
    rows = ""
    for k, v in sorted(dict_of_vars.items()):
        if type(v) in [int, str, float] and k[:2] != "__" or k in persist:
            # if type(v) not in ['function'] and k[:2] != "__":
            row = '''<tr>
            <td>{}</td><td>{}</td><td>{}</td>
            </tr>'''.format(k, v, str(type(v))[1:-1])
            # print(row)
            rows += row
            # if k in scenario_list:
            #     scenarios+=row
    return table.format(rows)

def write_report(vars, x, y, figs, scenario, mcmc=None):
    # Table of vars
    var_table = html_table(vars)
    scenarios_table = html_table(SCENARIO[scenario])
    persist = ['a_l', 'a_u', 'f', 'epsilon_ap', 'epsilon_wp']
    # scenario_list = ['alpha_ap', 'alpha_wp', 'omega_ap', 'omega_wp']

    # for k, v in sorted(vars.items()):
    #     if type(v) in [int, str, float] and k[:2] != "__" or k in persist:
    #         # if type(v) not in ['function'] and k[:2] != "__":
    #         row = '''<tr>
    #         <td>{}</td><td>{}</td><td>{}</td>
    #         </tr>'''.format(k, v, str(type(v))[1:-1])
    #         # print(row)
    #         var_table += row
    #         if k in scenario_list:
    #             scenarios+=row

    # Create dir
    _now = str(dt.now()).replace(':', '').replace("-", "").replace(" ", "_")[:-7]
    path = "./output/{}/".format(_now)
    # os.makedirs('./output/{}/'.format(_now))
    os.makedirs(path)

    # Create charts
    charts = ""
    for i, fig in enumerate(figs):
        fig.savefig(path + '/{}.jpg'.format(i))
        charts += '<img src="{}.jpg" width="100%" /><br/>\n'.format(i)
    if mcmc:
        fig, ax = plot_stoch_vars(mcmc)
        plt.tight_layout()
        fig.savefig(path + '/mcmc.jpg')
        charts += '<img src="mcmc.jpg" width="100%" /><br/>\n'
        # Write csv summary
        mcmc.write_csv(path+'/summary.csv', variables=[str(v) for v in mcmc.stochastics])

    html = '''
    <html>
    <head>
    <style>
    table {{
        border-collapse: collapse;
    }}
    table, th, td {{
        border: 1px solid black
    }}
    </style>
    <title>Report {current_time}</title>
    <body>
        {table}
        <p><b> Traces: {mcmc_traces} </b></p>
        <p>Scenario {scenario_num}
        {scenario}
        </p>
        <br/>
        {charts}
    </body>
    </html>'''.format(table=var_table,
                      mcmc_traces=mcmc.trace('omega')[:].shape[0],
                      scenario_num = scenario_number,
                      scenario=scenarios_table,
                      charts=charts,
                      current_time=_now)

    # Save files
    with open(path + "/index.html", 'w') as page:
        page.write(html)

    webbrowser.open(os.getcwd() + path + "/index.html")


    return path
