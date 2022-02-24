# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 12:52:24 2021

@author: Chang.Liu
"""

from dash import dash
import dash_bootstrap_components as dbc

## Diskcache
import diskcache
cache = diskcache.Cache("./cache")

from layout import app_layout
from callbacks import register_callbacks 

import sys
sys.path.insert(0, r'C:\Users\Chang.Liu\Documents\dev\Data_Importer')

# =============================================================================
# Dash app
# =============================================================================

# Running the server
if __name__ == "__main__":
    # View the app at http://192.168.2.77 or, in general,
    #   http://[host computer's IP address]
    
    app = dash.Dash(
        __name__,
        meta_tags=[{"name": "viewport", 
                    "content": "width=device-width, initial-scale=1"}],
        # long_callback_manager=long_callback_manager,
        # prevent_initial_callbacks=True,
        external_stylesheets=[dbc.themes.BOOTSTRAP]
    )
    app.config.suppress_callback_exceptions = True
    app.layout = app_layout
    register_callbacks(app)
    
    app.run_server(debug=False, host='0.0.0.0', port=80, use_reloader=False)
    # app.run_server(debug=True, port=80, dev_tools_silence_routes_logging = True)
    # app.run_server(debug=False, port=80, dev_tools_silence_routes_logging = True)
            
    
