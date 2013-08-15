from qtools import QtGui, QtCore
from collections import OrderedDict

# Generic classes
# ---------------
class TreeItem(object):
    def __init__(self, parent=None, data=None):
        """data is an OrderedDict"""
        self.parent_item = parent
        self.index = QtCore.QModelIndex()
        self.children = []
        # by default: root
        if data is None:
            data = OrderedDict(name='root')
        self.item_data = data
    
    def appendChild(self, child):
        self.children.append(child)
    
    def removeChild(self, child):
        self.children.remove(child)

    def removeChildAt(self, row):
        self.children.pop(row)
        
    def insertChild(self, child, index):
        self.children.insert(index, child)
        
    def child(self, row):
        return self.children[row]
        
    def rowCount(self):
        return len(self.children)
        
    def columnCount(self):
        return len(self.item_data)

    def data(self, column):
        if column >= self.columnCount():
            return None
        return self.item_data.get(self.item_data.keys()[column], None)
        
    def row(self):
        if self.parent_item is None:
            return 0
        return self.parent_item.children.index(self)
        
    def parent(self):
        return self.parent_item
       
        
class TreeModel(QtCore.QAbstractItemModel):
    def __init__(self, headers):
        QtCore.QAbstractItemModel.__init__(self)
        self.root_item = TreeItem()
        self.headers = headers
        
    def add_node(self, item_class=None, item=None, parent=None, **kwargs):
        """Add a node in the tree.
        
        
        """
        if parent is None:
            parent = self.root_item
        if item is None:
            if item_class is None:
                item_class = TreeItem
            item = item_class(parent=parent, **kwargs)
        
        row = parent.rowCount()
        item.index = self.createIndex(row, 0, item)
        
        self.beginInsertRows(parent.index, row-1, row-1)
        parent.appendChild(item)
        self.endInsertRows()
        
        return item
        
    def remove_node(self, child, parent=None):
        if parent is None:
            parent = self.root_item
            
        row = child.row()
        self.beginRemoveRows(parent.index, row, row)
        parent.removeChild(child)
        self.endRemoveRows()
        
    # def move_node(self, child, parent_target, child_target=None):
        # row = child.row()
        # parent_source = child.parent()
        # if child_target is not None:
            # child_target_row = child_target.row()
        # else:
            # child_target_row = parent_target.rowCount()
        # canmove = self.beginMoveRows(parent_source.index, row, row,
            # parent_target.index, child_target_row)
        # if canmove:
            
            
            # if parent is None:
                # parent = self.root_item
            # if item is None:
                # if item_class is None:
                    # item_class = TreeItem
            # item = child._(parent=parent, **kwargs)
            
            # row = parent.rowCount()
            # item.index = self.createIndex(row, 0, item)
            
            
            # parent_target.insertChild(child_new, child_target_row)
            # if parent_target == parent_source:
                # if child_target_row < row:
                    # row += 1
                # parent_source.removeChildAt(row)
            # else:
                # parent_source.removeChild(child)
            # # child.parent_item = parent_target
            
            # self.endMoveRows()
    
    def get_descendants(self, parents):
        if type(parents) != list:
            parents = [parents]
        nodes = []
        for parent in parents:
            nodes.append(parent)
            if parent.children:
                nodes.extend(self.get_descendants(parent.children))
        return nodes
        
    def all_nodes(self):
        return self.get_descendants(self.root_item)
        
    def index(self, row, column, parent=None):
        if parent is None:
            parent = self.root_item.index
        if not self.hasIndex(row, column, parent):
            return QtCore.QModelIndex()
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        child_item = parent_item.child(row)
        if child_item:
            index = self.createIndex(row, column, child_item)
            child_item.index = index
            return index
        else:
            return QtCore.QModelIndex()

    def parent(self, item):
        if not item.isValid():
            return QtCore.QModelIndex()
        item = item.internalPointer()
        parent_item = item.parent()
        if (parent_item == self.root_item):
            return QtCore.QModelIndex()
        index = self.createIndex(parent_item.row(), 0, parent_item)
        parent_item.index = index
        return index

    def rowCount(self, parent=None):
        if parent is None:
            parent = QtCore.QModelIndex()
        if parent.column() > 0:
            return 0
        if not parent.isValid():
            parent_item = self.root_item
        else:
            parent_item = parent.internalPointer()
        return parent_item.rowCount()
        
    def columnCount(self, parent=None):
        if parent is None:
            parent = QtCore.QModelIndex()
        if not parent.isValid():
            return len(self.headers)
        return parent.internalPointer().columnCount()

    def data(self, index, role):
        if role != QtCore.Qt.DisplayRole:
            return None
        item = index.internalPointer()
        return item.data(index.column())
        
    def setData(self, index, data, role):
        return False
        
    def supportedDropActions(self): 
        return QtCore.Qt.MoveAction         

    def flags(self, index):
        if not index.isValid():
            return QtCore.Qt.ItemIsEnabled
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable | \
               QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsDropEnabled
            
    def mimeTypes(self):
        return ['text/xml']

    def mimeData(self, indexes):
        data = ",".join(set([str(index.internalPointer()) for index in indexes]))
        mimedata = QtCore.QMimeData()
        mimedata.setData('text/xml', data)
        return mimedata

    def dropMimeData(self, data, action, row, column, parent):
        parent_item = parent.internalPointer()
        target = parent_item
        sources = data.data('text/xml').split(',')
        self.drag(target, sources)
        return True

    def drag(self, target, sources):
        """
        
        To be overriden.
        
        """
        print "drag", target, sources