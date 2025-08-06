## Timings
Timing patterns due to solution needing to converge when there is little time to solve (i.e. call OOP, flop)
- If real-time solution is not nearly converged, occasionally act according to blueprint (mostly on early streets)


## Client side
Process detection:
- Give the client-side script an inconspicuous name

Resource usage detection (maybe impossible):
- Make sure resources don't spike on OCR usage
- Worst case, do OCR on a seperate device


## Strategy
Same frequency to take an early game tree action with tight confidence interval over several different time intervals? 
- Change amount of clusters every X (50k to 100k-ish?) amount of hands
- Quantize preflop 2-bet/3-bet frequencies to 0%, 50%, or 100%
- Quantize 4-bet+ frequencies to 5% intervals?
- Purify 3-bet+ sizes:
  - Compute blueprint with many 3-bet sizes
  - When playing according to the blueprint, keep 3-bet frequencies unchanged, but replace sizing with the most frequently used 3-bet size in that spot
  - When solving in real time, let hero only choose the 3-bet+ size most frequently used in the blueprint

Winrate too high?
- When a terminal action (fold/terminal call/terminal all-in) is suboptimal, choose that action at random with the frequency determined by the amount of
  EV estimated to be lost
- Choose target EV to lose based on frequency of these situations occuring (analyze hand history data) and the amount by which the winrate should be reduced
