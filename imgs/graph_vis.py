from pymol import cmd

pdb = '1gnx.pdb1'
pdb_code = pdb.split('.')[0]
chain = 'A'
distance = 6

# cartoon representation
cmd.load(pdb, pdb_code)
cmd.hide('everything', f'not chain {chain}')
cmd.show_as('cartoon', f'chain {chain}')
cmd.spectrum('resi', selection=f'chain {chain} and name CA')

# graph representation
cmd.load(pdb, 'graph')
cmd.hide('everything', f'(not chain {chain}) and graph or not polymer.protein')
cmd.hide('cartoon', 'graph')

cmd.select('nodes', f'graph and name CA and chain {chain} and polymer.protein')

myspace = {'resids': set(), 'subresids': set()}
cmd.iterate(f'graph and name CA and chain {chain}', 'resids.add(resi)', space=myspace)

collection_str = f'graph and chain {chain} and name CA and resi ' + '+'.join(myspace['resids'])
cmd.show('sphere', collection_str)
cmd.spectrum('resi', palette='rainbow', selection=f'graph and chain {chain} and name CA')

for resid in myspace['resids']:
   cmd.iterate(f'(graph and chain {chain} and name CA) within {distance} of (graph and chain {chain} and name CA and resi {resid})', 'subresids.add(resi)',
               space=myspace)
   
   for resid2 in myspace['subresids']:
       cmd.bond(f'graph and chain {chain} and name CA and resi {resid}', f'graph and chain {chain} and name CA and resi {resid2}')

#       dist = cmd.distance('dist', 
#                           f'graph and chain {chain} and name CA and resi {resid}',
#                           f'graph and chain {chain} and name CA and resi {resid2}')


   myspace['subresids'] = set()

cmd.show('sticks', 'nodes')
cmd.set('stick_radius', -0.2)
cmd.set('sphere_scale', 0.5)

cmd.zoom(f'chain {chain}')

