from pertussis import *
import os
from datetime import datetime as dt

def write_report(vars, mcmc=None , x = None, y = None):
    # Table of vars
    _zzz = " "
    for k, v in sorted(vars.items()):
        if type(v) in [int,str, float] and k[:2] != "__":
            row = "\t<tr>\n\t\t<td>{}</td><td>{}</td>\n\t</tr>\n".format(k,v)
            # print(row)
            _zzz += row
    table = "<table>\n{}</table>".format(_zzz)

    # Create dir
    _now = str(dt.utcnow()).replace(':', '').replace("-","").replace(" ","_")[:-7]
    path = "./output/{}/".format(_now)
    # os.makedirs('./output/{}/'.format(_now))
    os.makedirs(path)

    # Create charts
    fig, ax = draw_model(x, y[3:], ["Infected Is", "Infected Ia", "Recovered", "Healthy", "All"], split=0, collapse=True)
    plt.tight_layout()
    fig.savefig(path+'/0.jpg')
    charts = ""
    for i in range(1):
            charts += '<img src="{}.jpg" width="100%" /><br/>\n'.format(i)

    #
    #  save to file
    html='''
    <html>
    <head>
    <style>

    <title>
    </title>
    <body>
        {table}
        {mcmc_traces}
        <br/>
        {charts}
    </body>
    </html>'''.format(table=table,
                      mcmc_traces="",
                      charts=charts)

    # Save files
    with open(path+"/index.html", 'w') as page:
        page.write(html)