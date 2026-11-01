###modified from https://github.com/cgoliver/RNAmigos/blob/master/data_processor/build_dataset.py

"""
    Takes ligand residue IDs from lig_dict (built by `binding_pocket_filter.py`)
    and builds a networkx binding site graph for each ligand.
"""

import os
import pickle
import itertools
from types import SimpleNamespace
import numpy as np
import networkx as nx
from Bio.PDB import *
from tqdm import tqdm
import logging
from random import shuffle



faces = ['W', 'S', 'H']
orientations = ['C', 'T']
valid_edges = set(['B53'] + [orient + "".join(sorted(e1 + e2)) for e1, e2 in itertools.product(faces, faces) for orient in orientations])

def remove_self_loops(G):
    selfs = list(nx.selfloop_edges(G))
    G.remove_edges_from(selfs)

def remove_non_standard_edges(G):
    remove = []
    for n1, n2, d in G.edges(data=True):
        if d['label'] not in valid_edges:
            remove.append((n1, n2))
    G.remove_edges_from(remove)

def symmetric(label):
    """
        Returns symmetric version of label.
        e.g symmetric(CHW) = CHW, symmetric(CWH) = CHW
    """
    #if not a base pair keep the same
    if label[0] not in ['C', 'T']:
        return label
    else:
        return label[0] + "".join(sorted(label[1:]))

def to_orig(G):
    """
        Convert new networkx graphs (from vlad 2019) to original format.
        Keep only B53 and 12 base pair interactions.
        Annotate nodes with position, chain, and nucleotide ID attributes.
        Arguments:
            G (networkx graph): networkx graph to convert to original.
        Returns:
            networkx graph
    """
    remove_self_loops(G)
    # remove_non_standard_edges(G)
    H = nx.Graph()
    for n1, n2, d in G.edges(data=True):
        label = symmetric(d['label'])
        if label in valid_edges:
            H.add_edge(n1, n2, label=label)

    #add pdb position and nt to node data
    d_orig = {n:d['nucleotide'].pdb_pos for n,d in G.nodes(data=True)}
    nx.set_node_attributes(H, {n:d_orig[n] for n in H.nodes()}, 'pdb_pos')
    d_orig = {n:d['nucleotide'].nt for n,d in G.nodes(data=True)}
    nx.set_node_attributes(H, {n:d_orig[n] for n in H.nodes()}, 'nt')
    d_orig = {n:n[0] for n,d in G.nodes(data=True)}
    nx.set_node_attributes(H, {n:d_orig[n] for n in H.nodes()}, 'chain')
    return H

def kill_islands(G, min_size=4):
    """
        Kill components that are too small
    """
    remove = []
    for comp in nx.connected_components(G):
        if len(comp) < min_size:
            remove.extend(list(comp))
        pass
    print(f"Removing {len(remove)} nodes in small components: {remove}")
    G.remove_nodes_from(remove)
def graph_ablations(G, mode):
    """
        Remove edges with certain labels depending on the mode.
        :params
        
        :G Binding Site Graph
        :mode how to remove edges ('bb-only', 'wc-bb', 'no-stack')
        :returns: Copy of original graph with edges removed/relabeled.
    """

    H = nx.Graph()

    H.add_nodes_from(G.nodes(data=True))
    # nx.set_node_attributes(H, 'pdb_pos', {n:d['pdb_pos'] for n,d in G.nodes(data=True)})
    # nx.set_node_attributes(H, 'nt', {n:d['nt'] for n,d in G.nodes(data=True)})
    if mode == 'label-shuffle':
        #assign a random label from the same graph to each edge.
        labels = [d['label'] for _,_,d in G.edges(data=True)]
        shuffle(labels)
        for n1, n2, d in G.edges(data=True):
            H.add_edge(n1, n2, label=labels.pop())
        return H

    if mode == 'no-label':
        for n1, n2, d in G.edges(data=True):
            H.add_edge(n1, n2, label='X')
        return H
    if mode == 'wc-bb-nc':
        for n1, n2, d in G.edges(data=True):
            label = d['label']
            if d['label'] not in ['CWW', 'B53']:
                label = 'NC'
            H.add_edge(n1, n2, label=label)
        return H

    if mode =='bb-only':
        valid_edges = ['B53']
    if mode == 'wc-bb':
        valid_edges = ['B53', 'CWW']

    for n1, n2, d in G.edges(data=True):
        if d['label'] in valid_edges:
            H.add_edge(n1, n2, label=d['label'])
    return H

def find_node(graph, chain, pos):
    for n,d in graph.nodes(data=True):
        if (n[0] == chain) and (d['nucleotide'].pdb_pos == str(pos)):
            return n
    return None

def has_NC(G):
    for n1, n2, d in G.edges(data=True):
        if d['label'] not in ['CWW', 'B53']:
            return True
    return False

def bfs_expand(G, initial_nodes, depth=2):
    """
        Extend motif graph starting with motif_nodes.
        Returns list of nodes.
    """

    total_nodes = [list(initial_nodes)]
    for d in range(depth):
        depth_ring = []
        for n in total_nodes[d]:
            for nei in G.neighbors(n):
                depth_ring.append(nei)
        total_nodes.append(depth_ring)
    return set(itertools.chain(*total_nodes))

def floaters(G):
    """
    Try to connect floating base pairs. (Single base pair not attached to backbone).
    Otherwise remove.
    """
    deg_ok = lambda H, u, v, d: (H.degree(u) == d) and (H.degree(v) == d)
    floaters = []
    for u,v in G.edges():
        if deg_ok(G, u, v, 1):
            floaters.append((u,v))

    G.remove_edges_from(floaters)

    return G

def dangle_trim(G):
    """
    Recursively remove dangling nodes from graph.
    """
    is_backbone = lambda n,G: sum([G[n][nei]['label'] != 'B53' for nei in G.neighbors(n)]) == 0
    degree = lambda i, G, nodelist: np.sum(nx.to_numpy_matrix(G, nodelist=nodelist)[i])
    cur_G = G.copy()
    while True:
        dangles = []
        for n in cur_G.nodes:
            if (cur_G.degree(n) == 1 and is_backbone(n, cur_G)) or cur_G.degree(n) == 0:
                dangles.append(n)
        if len(dangles) == 0:
            break
        else:
            cur_G.remove_nodes_from(dangles)
            cur_G = cur_G.copy()
            print(f"Removed dangling nodes: {dangles}")
    return cur_G

def stack_trim(G):
    """
    Remove stacks from graph.
    """
    is_ww = lambda n,G: 'CWW' in [info['label'] for node,info in G[n].items()]
    degree = lambda i, G, nodelist: np.sum(nx.to_numpy_matrix(G, nodelist=nodelist)[i])
    cur_G = G.copy()
    while True:
        stacks = []
        for n in cur_G.nodes:
            if cur_G.degree(n) == 2 and is_ww(n, cur_G):
                #potential stack opening
                partner = None
                stacker = None
                for node, info in cur_G[n].items():
                    if info['label'] == 'B53':
                        stacker = node
                    elif info['label'] == 'CWW':
                        partner = node
                    else:
                        pass
                if cur_G.degree(partner) > 3:
                    continue
                partner_2 = None
                stacker_2 = None
                for node, info in cur_G[partner].items():
                    if info['label'] == 'B53':
                        stacker_2 = node
                    elif info['label'] == 'CWW':
                        partner_2 = node
                try:
                    if cur_G[stacker][stacker_2]['label'] == 'CWW':
                        stacks.append(n)
                        stacks.append(partner)
                except KeyError:
                    continue
        if len(stacks) == 0:
            break
        else:
            cur_G.remove_nodes_from(stacks)
            cur_G = cur_G.copy()
    return cur_G

def in_stem(G, u,v):
    non_bb = lambda G, n: len([info['label'] for node, info in G[n].items() if info['label'] != 'B53'])
    is_ww = lambda G, u, v: G[u][v]['label'] == 'CWW'
    if is_ww(G,u,v) and (non_bb(G,u) in (1, 2)) and (non_bb(G,v) in (1, 2)):
        return True
    return False



class Atom(object):
    def __init__(self, atom_type, atom_label, x, y, z):
        self.atom_type = atom_type
        self.atom_label = atom_label
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"{self.atom_type} {self.atom_label} {self.x} {self.y} {self.z}"


class Nucleotide(object):
    def __init__(self, pos, nt, real_nt, chemically_modified, pdb_pos, pdb_pos_ins):
        self.pos = pos
        self.nt = nt
        self.real_nt = real_nt
        self.chemically_modified = chemically_modified
        self.pdb_pos = pdb_pos
        self.pdb_pos_ins = pdb_pos_ins
        self.auth_seq_id = None
        self.atoms = []


    def add_atom(self, atom):
        if not isinstance(atom, Atom):
            logging.debug(f"Trying to insert non Atom object in strand {strand} \
                          nucleotide at pos {pos}.\nThe atom to be inserted is:\n{atom}")
            raise Exception("This is not an Atom object")
        self.atoms.append(atom)


    def __repr__(self):
        return f"{self.nt} {self.real_nt} {self.chemically_modified} {self.pdb_pos} {self.pdb_pos_ins} {self.atoms}"


class Strand(dict):

    def __init__(self, *args, **kwargs):#, name, entity_id, description):
        super().__init__()
        if kwargs:
            self.name = kwargs.get('name', '')
            self.description = kwargs.get('description', '')
            self.entity_id = kwargs.get('entity_id', '')


    def __setitem__(self, key, val):
        if not isinstance(val, Nucleotide):
            logging.debug(f"Trying to add a non Nucleotide object to Strand.\n{val}")
            raise Exception(f"Trying to add a non Nucleotide object to RNA_Strand.\n{val}")
        if not isinstance(key, int):
            logging.debug(f"Trying to add a Nucleotide with non-integer key to strand\n{key}")
            raise Exception(f"Trying to add a Nucleotide with non-integer key to strand\n{key}")
        super().__setitem__(key, val)


class RNA_Molecule(dict):

    def __init__(self, pdb_id, title):
        self.pdb_id = pdb_id 
        self.title = title
        self.fr3d_graph = None
        super().__init__()


    def __setitem__(self, key, val):
        if not isinstance(val, Strand):
            logging.debug(f"Trying to add a non Strand object to RNA_Molecule.\n{val}")
            raise Exception(f"Trying to add a non Strand object to RNA_Molecule.\n{val}")
        super().__setitem__(key, val)

def _clean(s):
        return s.strip().strip('"').strip("'")
    

def parse_fr3d(fr3d_path, pdb_path):
    """
        Parse fr3d annotations and return a graph.
        Sample line:
        "1_A_17_G","s53","1_A_16_G"
        "1FJG|1|A|G|9","cWW","1FJG|1|A|C|25"
    """
    G = nx.Graph()
    nucs = {}

    with open(fr3d_path, "r") as fr3d:
        for line in fr3d:
            base_1, label, base_2 = line.split(",")

            b1 = _clean(base_1)
            b2 = _clean(base_2)
            t1 = b1.split("|")
            t2 = b2.split("|")

            if len(t1) != 5 or len(t2) !=5:
                continue

            chain_1, nuc_1, pos_1 = t1[2], t1[3], t1[4]
            chain_2, nuc_2, pos_2 = t2[2], t2[3], t2[4]

            nuc_1_clean = str(nuc_1).strip().strip('"').strip("'")
            nuc_2_clean = str(nuc_2).strip().strip('"').strip("'")
            if len(nuc_1_clean) > 1 or len(nuc_2_clean) > 1:
                continue

            label = label.strip().strip('"').strip("'").upper()
            if label in valid_edges:
                label = label[0] + "".join(sorted(label[1:]))
                G.add_edge((chain_1, pos_1), (chain_2, pos_2), label=label)
                nucs[(chain_1, str(pos_1))] = nuc_1[0].strip()
                nucs[(chain_2, str(pos_2))] = nuc_2[0].strip()
    
    nx.set_node_attributes(G, nucs, 'nuc')

    # use BioPython to fill in the backbones
    parser = MMCIFParser(QUIET=True)
    struc = parser.get_structure('', pdb_path)[0]
    for chain in struc:
        reslist = list(chain)
        for i,residue in enumerate(reslist):
            if i == 0:
                continue
            r1 = (chain.id, str(residue.id[1]))
            r2 = (chain.id, str(reslist[i-1].id[1]))
            # print(r1, r2)
            try:
                G[r1]
                G[r2]
            except:
                continue
            else:
                G.add_edge(r1, r2, label='B53')
    
    for n, d in list(G.nodes(data=True)):
        nt = d.get('nuc', None)
        pdb_pos = n[1]
        chain_id = n[0]
        G.nodes[n]['nucleotide'] = SimpleNamespace(pdb_pos=str(pdb_pos), nt=(nt if nt else None))
        G.nodes[n]['pdb_pos'] = str(pdb_pos)
        G.nodes[n]['nt'] = (nt if nt else None)
        G.nodes[n]['chain'] = chain_id
    return G


def lig_center(lig_atoms):
    return np.mean(np.array([a.coord for a in lig_atoms]), axis=0)

def find_residue(chain, pos):
    """
        Get BioPython residue object from chain and position.
        Arguments:
            chain (BioPython.Chain): chain object
            pos (int): position on chain
        Returns:
            Bio.PDB.Residue: residue object matching position.
    """
    for residue in chain:
        if residue.id[1] == pos:
            return residue
    return None

def graph_from_residues(full_graph, residues, expand_depth=0):
    """
        Build an NX graph from list of biopython residues.
        Arguments:
            full_graph (Networkx graph): networkx graph of full RNA structure.
            residues (list): list of BioPython Residues to get graph for.
            expand_depth (int): number of hops in graph to extend pocket by.
        Returns:
            Networkx graph: networkx graph with nodes matching `residuses`
    """
    pocket_nodes = []
    for r in residues:
        node = find_node(full_graph, r.get_parent().id, r.id[1])
        if node is not None:
            pocket_nodes.append(node)
    if expand_depth:
        pocket_nodes = bfs_expand(full_graph, pocket_nodes, depth=expand_depth)

    pocket_graph = full_graph.subgraph(pocket_nodes).copy()
    G = to_orig(pocket_graph)
    print(f"Initial pocket graph has {len(G.nodes())} nodes and {len(G.edges())} edges.")
    #remove bases with no connections
    kill_islands(G)
    #remove dangles (only backbone interactions not in loops)
    G = dangle_trim(G)
    kill_islands(G)

    print(f"After kill_islands and dangle trim has {len(G.nodes())} nodes and {len(G.edges())} edges.")
    return G

def get_pocket_graph(pdb_structure_path, ligand_id, graph,
        ablate=None, dump_path="data/nx/", cutoff=10,
        non_binding=False, max_non_bind_samples=5):
    """
        Main function for extracting a graph from a binding site.
        Dumps graph representing the pocket around given ligand in PDB.
        Arguments;
            pdb_structure_path (str): path to full PDB structure
            ligand_id (str): string describing ligand position in PDB
            graph (Networkx): full graph of RNA structure
            ablate (str): which ablation to perform on final graph (see `graph_process.py`)
            dump_path (str): path to write binding site graph
            cutoff (int): Maximum distance from ligand to include for graph residues. (default=10)
            non_binding (bool): If True, samples non-binding sites for each binding site (default=False).
            max_non_bind_sample (int): Number of non-binding sites to sample. (default=5)
        Returns:
            networkx graph: graph representing binding site around ligand.
    """
    #load PDB
    print(ligand_id)
    parser = MMCIFParser(QUIET=True)
    pdbid = os.path.basename(pdb_structure_path).split(".")[0]
    structure = parser.get_structure("", pdb_structure_path)[0]

    chain,resname, pos = ligand_id.split(":")[1:]

    #find ligand residue and get its center coordinates
    lig_residue = find_residue(structure[chain], int(pos))
    lig_res_atoms = lig_residue.get_atoms()
    lig_coord = lig_center(lig_res_atoms)

    #get atoms within radius
    kd = NeighborSearch(list(structure.get_atoms()))
    pocket = kd.search(lig_coord, cutoff, level='R')


    G = graph_from_residues(graph, pocket, expand_depth=1)

    labels = {d['label'] for _,_,d in G.edges(data=True)}

    assert labels.issubset(valid_edges)

    if len(G.nodes()) < 4:
        return None

    if dump_path:
        outp = os.path.join(dump_path, f"{pdbid}_{ligand_id}_BIND.nx")
        with open(outp, "wb") as fh:
            pickle.dump(G, fh)
    return G 

def get_binding_site_graphs_all(lig_dict_path, dump_path, non_binding=False):
    """
        Get binding site graphs for each ligand in lig_dict.
        lig_dict is a dictionary with {'pdbid': [ligand_residues]} generated by `binding_sites.py`
        Saves a binding pocket graph for each ligand.
        Args:
            lig_dict_path (str): Path to pdb ligand annotations dictionary (see `binding_sites.py`)
            dump_path (str): Path to folder where annotated graphs will be dumped.
            non_binding (bool): If `True` will sample non-binding sites which can be used for binding
                                site finding.
    """

    lig_dict = pickle.load(open(lig_dict_path, 'rb'))

    print(f">>> building graphs for {len(lig_dict)} PDBs")
    print(f">>> dumping in {dump_path}")
    print(f">>> and {sum(map(len, lig_dict.values()))} binding sites.")

    failed = 0

    try:
        os.mkdir(dump_path)
    except:
        pass

    done_pdbs = {f.split('_')[0] for f in os.listdir(dump_path)}
    print(f">>> skipping {len(done_pdbs)}")

    failed = []
    empties = 0
    num_found = 0
    missing_graphs = []
    for pdbid, ligs in tqdm(lig_dict.items()):
        pdbid =  pdbid.split(".")[0]

        pdb_path = f"data/pdb.cif/{pdbid}.cif"
        fr3d_path=f"data/pdb_fr3d_basepairs/{pdbid}_fr3d_basepairs.csv"
        if pdbid in done_pdbs:
            continue
        # try:
        print(">>> ", pdbid)
        try:
            pdb_graph=parse_fr3d(fr3d_path, pdb_path)
        except FileNotFoundError:
            print(f"{pdbid} graph not found.")
            print(f"{pdbid} had {len(ligs)} binding sites")
            missing_graphs.append(pdbid)
            continue
        try:
            pdb_graph.nodes()
        except AttributeError:
            print("empty graph")
            continue
        # print(f"new guy: {pdbid}")
        # continue
        for lig in ligs:
            #dump binding site graphs
            try:
                g = get_pocket_graph(pdb_path, lig,
                                pdb_graph, dump_path=dump_path,
                                non_binding=non_binding)
                if g is None:
                    empties += 1
                else:
                    num_found += 1
                    print(f">>> pockets so far {num_found}")
                    
            except FileNotFoundError:
                print(f"{pdbid} not found")
                failed.append(pdbid)
    print(f">>> missing graphs for {missing_graphs}")
    print(f">>> failed on {len(failed)} graphs")
    print(f">>> got {empties} empty graphs")


if __name__ == "__main__":
    get_binding_site_graphs_all("data/binding_pocket/binding_pocket_dict_filter.p", 
                                "data/nx", non_binding=False)
    pass
