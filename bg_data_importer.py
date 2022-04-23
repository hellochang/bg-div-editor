# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 14:39:20 2020

@author: Todd.Liu
"""

import os
import pandas as pd
import numpy as np
import fredapi # quandl,
import datetime
from turbodbc import connect, make_options
from turbodbc import Megabytes

class DataImporter(object):

    def __init__(self,
                 verbose=True,
                 fsyms=None,
                 benchmark=None,
                 exchanges=None,
                 rates=None,
                 sectors=None,
                 port_ids=None):

        self.command_list = ['adjusted price', 'security info', 'rate'
                             'holding', 'dividend', 'benchmark price',
                             'historic benchmark constituent', 'dividend']
        self.frequently_used_identifiers = \
            ['benchmark', 'exchanges', 'bm_constituents',
             'rate_names', 'portfolios', 'us_tdates', 'sectors']
        if fsyms is not None: self.fsyms = list(fsyms)
        if benchmark is not None: self.benchmark = list(benchmark)
        if exchanges is not None: self.exchanges = list(exchanges)
        if rates is not None: self.rates = list(rates)
        if sectors is not None: self.sectors = list(sectors)
        if port_ids is not None: self.port_ids = list(port_ids)
        self.verbose = verbose
        # self._set_up_db_connection()
        # self._get_frequently_used_groups()
        self.db_datatypes = ['nvarchar', 'date', 'float', 'int', 'decimal']
        self.rate_names = {"FRED": ["DTWEXM",
                                    "TEDRATE",
                                    "DTWEXAFEGS",
                                    "UNRATE",
                                    "U1RATE",
                                    "U6RATE",
                                    "BOPGSTB",
                                    "CPIAUCSL",
                                    "TB3MS",
                                    "GDP",
                                    "GNP"
                        			]
                        }
        # print("Data Importer is initiated")
        if verbose:
            print('To import pre-defined data. Please select from one of the '
                  'following commands:')
            for s in self.command_list:
                print("\t|    '" + s + "'")
            print("e.g: load_pre_defined_data('adjusted price', self.bm_constituents)")
            print("e.g: load_pre_defined_data('rate','TB3MS')")
            print('')
            print("To load self-defined data. Please use the command:")
            print("e.g: load_data(query)")


    def __enter__(self):
            """
            Magic function for self referening within the "With Statement".


            Returns
            -------
            dbEngine
                Bascially return `self` so the calss would be usable on the fly.
            """
            return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Magic function for self-destruction used within the "With Statement".


        Returns
        -------
        None
        """
        del self


    # def __del__(self):
        # if self.connection: self.connection.close()
    def _get_frequently_used_groups(self):
        """
        Get frequently used identifiers such as benchmark constituents.

        Returns
        -------
        None.

        """
        def setup_query(query_name, identifier=None):
            if identifier is not None:
                identifier = self._add_list_to_query(identifier)
            query = "exec fstest.dbo.usp_GetFrequentlyUsedGroups "\
                f"'{query_name}','{identifier}'"
            return query

        self.benchmark = ['sp500', 'sptsx']
        self.exchanges = ['nys', 'tsx']
        self.bm_constituents = \
            self.load_data(setup_query('benchmark_constituent', 'sp500'), True)
        self.rate_names = self.load_data(setup_query('rate'), True)
        self.portfolios = self.load_data(setup_query('portfolio'), True)
        self.us_tdates = \
            self.load_data(setup_query('trading dates', 'nys'), True)
        self.sectors = self.load_data(setup_query('sector'), True)
        # identifiers = "'" + "','".join(self.frequently_used_identifiers) + "'"
        if self.verbose:
            print('Frequently used identifiers have been loaded. \n'
                  'Please select one of the following class variables:')
            for s in self.frequently_used_identifiers:
                print("\t|    " + s)

    @staticmethod
    def _add_list_to_query(identifiers, query=None):
        """
        Returns a query that execute a stored procedure with
        parameters in correct format. Convert a list of string from Python
        format to SQL format for stored procedures.
        For example, convert ['a','b'] to '''a'',''b'''
        Parameters
        ----------
        identifiers : String or list of strings.

        query : String, optional
            Main component of stored procedure.

        Returns
        -------
        Query

        """

        if isinstance(identifiers, str): identifiers = [identifiers]
        list_str = "''" + "'',''".join(identifiers) + "''"
        if query is None: return list_str

        query = query.replace('\n','')
        if query[-1] in [';', ',']:
            query = query[:-1]
        list_str = ",'" + list_str + "'"
        return query + list_str

    def _set_up_db_connection(self):
        """
        Setup database connection using turbodbc package.

        Returns
        -------
        None.

        """
        options = make_options(read_buffer_size=Megabytes(42),
                               large_decimals_as_64_bit_types=True)
        if os.name == 'posix':
            conn_str = 'Driver={/usr/local/lib/libtdsodbc.so};'\
                       'Server=192.168.2.223;Port=1433;UID=extsql;'\
                       'PWD=60Bedford;TDS_Version=8.0'
            connection = connect(connection_string = conn_str,
                                 turbodbc_options=options)
        else:
            connection = connect(dsn='Dev2016;',
                                 turbodbc_options=options)
        self.connection = connection
        self.cursor = connection.cursor()

    def load_data(self, query, output_list=False):
        """
        Import data from database using the query provided,

        Parameters
        ----------
        query : string
            Query defaults to a stored procedure dedicated to the importer,
            but can be a user defined query with select statement.
        output_list : Bool, optional
            An option to return first column of the output dataframe as list.
            The default is False.

        Returns
        -------
        df : pd.DataFrame
            Result dataframe

        """
        try:
            self._set_up_db_connection()
            self.cursor.execute(query)
            df = self.cursor.fetchallarrow()
            df = df.to_pandas(ignore_metadata=True, date_as_object=False)
            # df = df.to_pandas()
            if output_list: return df.iloc[:,0].to_list()
        except Exception as e:
            print(f'ERROR: {e}')
            self.connection.close()
            return
        self.connection.close()
        return df

    def load_pre_defined_data(self, query_name, identifiers=None):
        """
        Fill in the parameters for stored procedure usp_DataImporter to load
        pre-define data.

        Parameters
        ----------
        query_name : String
            Types of data that the stored procedure supports. Available types
            are provided as a class property.
        identifiers : String, optional
            Conditions for some queries. For example, when querying adjhusted
            price only for selected fsym_ids, identifiers should be supplied
            with fsym_ids in strings. The default is None.

        Returns
        -------
        df : pd.DataFrame
            Result dataframe

        """
        query = f"exec fstest.dbo.usp_DataImporter '{query_name}'"
        query = self._add_list_to_query(identifiers, query)
        df = self.load_data(query)
        return df


    def insert(self, table_name, df=None):
        """
        Insert data into the database

        Parameters
        ----------
        table_name : String
            Table name must be written in the format of database.schema.table
        df : pd.DataFrame, optional

        Returns
        -------
        """


        #check if a dataframe is provided
        if df is None or (not isinstance(df, pd.DataFrame)) or (df.empty):
             raise Exception('Must supply a dataframe in order to insert data into the database.')
        db_name = table_name.split('.')[0]
        #check if proper name is provided
        if db_name.lower() not in ['fstest', 'development']:
            raise Exception('Must supple database name. \
                  Choose from fstest or development')

        table = self.load_data(f"""SELECT count(*)
                                  FROM {db_name}.information_schema.tables
                                  where TABLE_NAME = '{table_name.split('.')[0]}'
                              """)
        if table.empty:
            raise Exception(f"{table_name} table does not exist.")

        # prepare column name
        column_names = '(' + ', '.join(df.columns) + ')'
        # prepare values place holders
        values_filler = '(' + ', '.join(['?' for col in df.columns]) + ')'
        # combine the above with database command into sql query
        query = f"""
                    INSERT INTO {table_name} {column_names}
                    VALUES {values_filler};
                """
        # insert data
        #values = [df[col].values for col in df.columns]
        values = [np.ma.MaskedArray(df[col].values, pd.isnull(df[col].values)) for col in df.columns]
        print('prepared')
        # values = []
        # for col in df.columns:
        #     if df[col].dtype == np.float64:
        #         value = np.array([nan_to_null(v) for v in list(df[col].values)])
        #     else:
        #         value = df[col].values
        #     values.append(value)

        with self.cursor as cursor:
            try:
                cursor.executemanycolumns(query, values)
                self.connection.commit()
                print(f'Inserted into {table_name}.')
            except Exception:
                print(Exception)
                self.connection.rollback()
                print('something went wrong')
#%%
    def read_fred(self, items: list, start_date: str = '1995-01-01', end_date: str = None) -> list:
        """A wrapper for reading economic data provided by fred.

        Parameters
        ----------
        items: list
            List of codes for the data pull. These names should match FRED identifiers.

        Returns
        -------
        list
            dbs is returns as a list of data
        """
        #quandl.ApiConfig.api_key = "JJGeGRsBDph8bGLftM4s"
        fred = fredapi.Fred(api_key='fc7993c36368433ec620a0f30214e026')

        dbs = []
        if end_date is None:
            end_date = datetime.date.today().isoformat()
        for item in items:
            db = pd.DataFrame(fred.get_series(item,
                                              start_date= start_date, \
                                              end_date = end_date)).dropna()
            dbs.append(db)
        return dbs
