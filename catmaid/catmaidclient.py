import requests

class CatmaidClient:
    """  
    Barebones CATMAID API interface.
    Saves base url, credentials
    Use the fetch method to make API requests.

    Requires API Token:
    https://catmaid.readthedocs.io/en/stable/api.html#api-token

    Zhen Lab API reference:
    https://zhencatmaid.com/apis/

    For undocumented API resources:
    https://catmaid.readthedocs.io/en/stable/api.html

    ...

    Attributes
    ----------
    server : str
        server url
    api_token : str
        api_token
    project_id : int
        CATMAID project ID

    Methods
    -------
    fetch
        HTTP request to CATMAID server
    
    call
        Print server URL
    """

    def __init__(self,base_url,api_token=None):
        """
        Parameters
        ----------
        server : str
            server url
        api_token : str
            api token
        """        
        if type(base_url) != str:
            raise TypeError("Invalid server URL, must be string")
        
        self.base_url = base_url
        self.api_token = api_token

    def fetch(self,url,method,data=None):
        """ Make HTTP request to CATMAID server   
        
        Parameters
        ----------
        url : str
            URL to send the request to
        method : str
            "get" or "post"
        data : dict, optional
            Parameters or data to be included in request 

        Returns
        -------
        Response object
            HTTP Response object from 'requests' package
            
        Notes
        -------
        Use Response.text() or Response.json() to see response
        """
        auth_header = {'X-Authorization': 'Token ' + self.api_token}

        if method == "get":
            result = requests.get(url = self.base_url + url,
                                    params = data,
                                    headers = auth_header)
        if method == "post":
            result = requests.post(url = self.base_url + url,
                                   data = data,
                                   headers = auth_header)
        return result
    
    def __call__(self):
        """ Print server url 
        """
        print("Server: " + self.base_url)