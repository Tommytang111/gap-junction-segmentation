from catmaidclient import CatmaidClient

class CatmaidHelper(CatmaidClient):
    def __init__(self,base_url,api_token=None,**kwargs):
        super().__init__(base_url,api_token,**kwargs)

        if 'project_id' in kwargs:
            self.pid = kwargs['project_id']

    def set_project(self,pid):
        """ Set project ID
        """
        self.pid = pid

    def get_projects(self):
        """ Get projects available to user
        """
        return self.fetch(
            url = "/projects/",
            method = "get"
            ).json()
    
    def get_skeletons(self):
        """ Return list of skeleton IDs

        Parameters
        ----------
        project : CatmaidProject
            Referencing CatmaidProject.py

        Returns
        -------
        list
            List of skeleton IDs
        """
        return self.fetch(
            url = "/" + str(self.pid) + "/skeletons/",
            method = "get",
            data = {"project_id": self.pid}
            ).json()
    
    def load_skeleton_names(self,skeletons):
        """ Make dictionary associating skeleton ID and skeleton name

        Parameters
        ----------
        project : CatmaidProject
            Referencing CatmaidProject.py
        skeletons : list
            List of skeleton IDs

        Returns
        -------
        dict
            Dictionary {skeleton id : skeleton name}
        """
        data = {'neuronnames': '1',
                'metaannotations': '0'}

        for i in range(len(skeletons)):
            data["skeleton_ids['" + str(i) + "']"] = skeletons[i]

        return self.fetch(
            url = "/" + str(self.pid) + "/skeleton/annotationlist",
            method = "post",
            data = data
            ).json()['neuronnames']


if __name__ == "__main__":
    base_url = "https://zhencatmaid.com"

    f = open("C:/Users/ishaan/Desktop/william/wlapit.txt", "r")
    api_token = f.read().replace('\n', '')
    f.close()

    test = CatmaidHelper(base_url = base_url,
                            api_token = api_token)
    
    proj_dict = {}

    projects = test.get_projects()

    for project in projects:
        proj_dict[project['title']] = project["id"]

    print(proj_dict)
    