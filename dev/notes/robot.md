Spike sorting robot
===================

The goal of the robot is to guide the user in the manual stage of refinement
of the automatic step's output (KlustaKwik). Here is the cyclical workflow:

  * The robot selects a set of two or more clusters which seem similar 
    according to some metrics (to be defined).
  * The robot zooms in automatically in the relevant channels in the waveform
    view, and selects the best projection in the feature view 
    (projection which maximises the discrepancy between the clusters, e.g.
    projection pursuit).
  * The user is asked whether these clusters should be merged or not. He can
    navigate in the klustaviewa interface as usual (waveform overlay, highlighting,
    changing projections, etc.), and then clicks on YES or NO.
  * The clusters are merged or not, and the robot goes to the next proposition.


Merge propositions
------------------

We define a **merge proposition** as a set of $k$ clusters ($k \geq 2$) that 
are similar and candidates for a merge operation. Several metrics are 
associated to a merge proposition (cross-correlation, etc.). The robot 
automatically computes a list of propositions when the data set is loaded in 
KlustaViewa, and sorts them in decreasing order of confidence (i.e. the first 
proposition is likely to be True (merge), whereas the last proposition is 
more ambiguous).

There may be a problem when several propositions concern the same cluster, 
since the clusters may change between the first and the second proposition 
if a merge operation happens. The robot should detect these cases and 
recompute the relevant propositions whenever a merge operation occurs.


Robot widget
-------------

There is a special widget in the interface that offers a way for the user
to interact with the robot. This widget contains:

  * The list of all propositions (for each proposition, the list of clusters
    and some metrics information).
  * Previous and Next buttons for navigating through the list of propositions.
  * Yes or No buttons for deciding whether to merge the clusters or not.
  
At each proposition, the user looks at the data, then selects Yes or No. If he
chooses Yes, he's given a chance to look at the result and cancel the 
proposition. Then, he can select Next to go to the next proposition.


Split operation
---------------

Splitting is also possible at any time. The idea of the robot is to be fully
integrated in the interface so that the typical, non-linear workflow permitted
by the interface is still possible. The only non-trivial point is that
merge and split operations should trigger an update of the robot's list of 
propositions.


Robot's implementation notes
----------------------------

There are two independent modules: the robot widget, and the robot 
intelligence (i.e. the algorithms). The widget implements the robot 
interface, and a way to communicate with the algorithms. 

  * The widget requests the AI a list of propositions in the current data
    state.
  * The widget informs the AI about a merge decision, so that the AI can
    update the list of propositions.
  * Idem for a split decision.
  
Example of an AI interface:

    class RobotAI(object):
        def __init__(self, dh, sdh):
            """Constructor. Accepts the data holder and select data holder
            as arguments."""
            self.dh = dh
            self.sdh = sdh
    
        def propositions(self):
            """Return the list of merge propositions.
            
            Each proposition is a dict with the following keys:
              * clusters: list of cluster indices to be merged
              * ...
            
            """
            return []
        
        def merge_occurred(self, merged):
            """Called when a merge operation happens.
            
            Arguments:
              * merged: a list of cluster indices.
            
            """
        
        def split_occurred(self, split, clusters):
            """Called when a split operation happens.
            
            Arguments:
              * split: a list of cluster indices (pre-split numbering)
              * clusters: the new assignement of clusters for every spike.
            
            """
        
A particular robot algorithm is defined by creating a new class deriving
from `RobotAI`.


Things to do
------------

  * Implement merge and split operations in the interface, along with a
    undo/redo stack. This is totally independent from the robot idea.
  * Implement the robot widget which interfaces with a robot "artificial
    intelligence". The interface can be implemented independently from
    the specific algorithm as soon as the common interface is decided
    (i.e. the `RobotAI` interface).
  * Implement the automatic waveform zoom-in and feature projection.
  * Choose the metrics of interest that allow to define and sort the merge
    propositions, and implement a non-trivial robot!



New notes
---------

  * Sort all clusters by decreasing quality (amplitude, spike count, etc.)
  * Take the first cluster, and sort all neighbor clusters by decreasing
    order, according to some metric
  * Ask the user whether the two clusters should be merged: yes, no, 
    next cluster
  * Go to next neighbor, until the user decides to go to the next cluster
  * Repeat


