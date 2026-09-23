"""domibot 2: PPO self-play, reusing the domibot engine and training.encoding
unchanged from the MCTS lineage (training/, see its README) -- a different
training algorithm on the same game, not a different game.

See training/README.md's "domibot 2" section and the approved design plan
for why: MCTS self-play with a Monte-Carlo (or short TD-bootstrapped)
value target never once discovered engine play across five separate
conditions, while pure self-play's own equilibrium dynamics discouraged
crossing the "half-built engine looks worse than tuned Big Money" valley.
PPO's GAE gives dense, bootstrapped credit to every decision from a proper
value function, and needs no tree search at data-generation time.
"""
