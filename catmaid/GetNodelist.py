# obtain list of nodes in specified boundary

import numpy as np

def compile_tag_list(tag,tag_info):
    """ return list of nodes with tag

    Parameters
    ----------
    tag : str
        tag to search for
    tag_info : list
        list of nodes associated with tags [[id1,tag1],[id2,tag2]...]

    Returns
    -------
    list
        List of nodes associated with tag
    """
    nodes = []
    tag_indices = [(i, skeleton.index(tag))
                     for i, skeleton in enumerate(tag_info)
                     if tag in skeleton]
    if not tag_indices:
        pass
    else:
        for i in tag_indices:
            nodes.append(tag_info[i[0]][0])
    return nodes

def get_children(tree_info,node):
    """ return node IDs of children of reference node

    Parameters
    ----------
    tree_info : numpy array
        2D array of node IDs with 2 columns ['node ID', 'parent ID']
    node : int
        node id

    Returns
    -------
    list
        List of node IDs of children of reference node.
        If reference node has no children, returns 0
    """
    nodelist = []
    node_search = np.where(tree_info[:,1] == node)[0]
    if node_search.size != 0:
        for i in node_search:
            nodelist.append(tree_info[i][0])
        return nodelist
    else:
        return 0

def get_bounded_nodes(project,
                      skid,
                      start_tag="nerve_ring_starts",
                      end_tag="nerve_ring_ends"):
    """ 
    Returns a list of node IDs derived from the skeleton subtree
    bounded by the start_tag and the end_tag

    Parameters
    ----------
    project : CatmaidProject
        Referenced from CatmaidProject.py
    skid : int
        skeleton ID
    start_tag : str
        start tag text
    end_tag : str
        end tag text

    Returns
    -------
    dict
        If any nodes exist between start and end tag, return node IDs
    """
    url = str(project.pid) + "/skeletons/" + str(skid) + "/node-overview"
    skeleton_info = project.fetch(url = url,
                                    method = "get",
                                    data = {'project_id': project.pid,
                                            'skeleton_id': skid}).json()

    start_nodes = compile_tag_list(start_tag,skeleton_info[2])
    end_nodes = compile_tag_list(end_tag,skeleton_info[2])

    if start_nodes:
        tree_info = np.array(skeleton_info[0])[:,:2]

        nodelist = []

        #append nodes to nodelist from start tag to end tag or leafnode
        for start_node in start_nodes:
            checklist = [start_node]
            while len(checklist) > 0:
                for node in checklist:
                    nodelist.append(node)
                    if (node in end_nodes):
                        pass
                    elif get_children(tree_info,node):
                        checklist = checklist + (get_children(tree_info,node))
                    checklist.remove(node)
        return {'skid': skid, 'nodelist': nodelist}
    else:
        return None