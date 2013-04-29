







class FeatureWidget(VisualizationWidget):
    def create_view(self, dh):
        self.dh = dh
        self.view = FeatureView(getfocus=False)
        self.view.set_data(fetdim=self.dh.fetdim,
                      features=self.dh.features,
                      clusters=self.dh.clusters,
                      clusters_ordered=self.dh.clusters_ordered,
                      nchannels=self.dh.nchannels,
                      nextrafet=self.dh.nextrafet,
                      # colormap=self.dh.colormap,
                      cluster_colors=self.dh.cluster_colors,
                      masks=self.dh.masks,
                      # spike_ids=self.dh.spike_ids,
                      # spikes_rel=self.dh.spikes_rel,
                      )
        return self.view
        
    def update_view(self, dh=None):
        if dh is not None:
            self.dh = dh
        self.view.set_data(fetdim=self.dh.fetdim,
                      features=self.dh.features,
                      clusters=self.dh.clusters,
                      nchannels=self.dh.nchannels,
                      nextrafet=self.dh.nextrafet,
                      cluster_colors=self.dh.cluster_colors,
                      clusters_unique=self.dh.clusters_unique,
                      clusters_ordered=self.dh.clusters_ordered,
                      masks=self.dh.masks,
                      spike_ids=self.dh.spike_ids,
                      spikes_rel=self.dh.spikes_rel,
                      )
        self.update_nspikes_viewer(self.dh.nspikes, 0)
        self.update_feature_widget()

    def create_toolbar(self):
        toolbar = QtGui.QToolBar(self)
        toolbar.setObjectName("toolbar")
        toolbar.setIconSize(QtCore.QSize(32, 32))
        
        # navigation toolbar
        toolbar.addAction(klustaviewa.get_icon('hand'), "Move (press I to switch)",
            self.set_navigation)
        toolbar.addAction(klustaviewa.get_icon('selection'), "Selection (press I to switch)",
            self.set_selection)
            
        # toolbar.addSeparator()
            
        # autoprojection
        # toolbar.addAction(klustaviewa.get_icon('hand'), "Move (press I to switch)",
            # self.main_window.autoproj_action)
        
        toolbar.addSeparator()
        
        return toolbar
        
    def initialize_connections(self):
        ssignals.SIGNALS.ProjectionChanged.connect(self.slotProjectionChanged, QtCore.Qt.UniqueConnection)
        ssignals.SIGNALS.ClusterSelectionChanged.connect(self.slotClusterSelectionChanged, QtCore.Qt.UniqueConnection)
        ssignals.SIGNALS.HighlightSpikes.connect(self.slotHighlightSpikes, QtCore.Qt.UniqueConnection)
        ssignals.SIGNALS.SelectSpikes.connect(self.slotSelectSpikes, QtCore.Qt.UniqueConnection)
        
    def slotHighlightSpikes(self, sender, spikes):
        self.update_nspikes_viewer(self.dh.nspikes, len(spikes))
        if sender != self.view:
            self.view.highlight_spikes(spikes)
        
    def slotSelectSpikes(self, sender, spikes):
        self.update_nspikes_viewer(self.dh.nspikes, len(spikes))
        
    def slotClusterSelectionChanged(self, sender, clusters):
        self.update_view()
        
    def slotProjectionChanged(self, sender, coord, channel, feature):
        """Process the ProjectionChanged signal."""
        
        if self.view.data_manager.projection is None:
            return
            
        # feature == -1 means that it should be automatically selected as
        # a function of the current projection
        if feature < 0:
            # current channel and feature in the other coordinate
            ch_fet = self.view.data_manager.projection[1 - coord]
            if ch_fet is not None:
                other_channel, other_feature = ch_fet
            else:
                other_channel, other_feature = 0, 1
            fetdim = self.dh.fetdim
            # first dimension: we force to 0
            if coord == 0:
                feature = 0
            # other dimension: 0 if different channel, or next feature if the same
            # channel
            else:
                # same channel case
                if channel == other_channel:
                    feature = np.mod(other_feature + 1, fetdim)
                # different channel case
                else:
                    feature = 0
        
        # print sender
        log_debug("Projection changed in coord %s, channel=%d, feature=%d" \
            % (('X', 'Y')[coord], channel, feature))
        # record the new projection
        self.projection[coord] = (channel, feature)
        
        # prevent the channelbox to raise signals when we change its state
        # programmatically
        self.channel_box[coord].blockSignals(True)
        # update the channel box
        self.set_channel_box(coord, channel)
        # update the feature button
        self.set_feature_button(coord, feature)
        # reactive signals for the channel box
        self.channel_box[coord].blockSignals(False)
        
        # update the view
        self.view.process_interaction('SelectProjection', 
                                      (coord, channel, feature))
        
    def set_channel_box(self, coord, channel):
        """Select the adequate line in the channel selection combo box."""
        self.channel_box[coord].setCurrentIndex(channel)
        
    def set_feature_button(self, coord, feature):
        """Push the corresponding button."""
        if feature < len(self.feature_buttons[coord]):
            self.feature_buttons[coord][feature].setChecked(True)
        
    def select_feature(self, coord, fet=0):
        """Select channel coord, feature fet."""
        # raise the ProjectionToChange signal, and keep the previously
        # selected channel
        ssignals.emit(self, "ProjectionToChange", coord, self.projection[coord][0], fet)
        
    def select_channel(self, channel, coord=0):
        """Raise the ProjectionToChange signal when the channel is changed."""
        # print type(channel)
        # return
        # if isinstance(channel, basestring):
        if channel.startswith('Extra'):
            channel = channel[6:]
            extra = True
        else:
            extra = False
        # try:
        channel = int(channel)
        if extra:
            channel += self.dh.nchannels #* self.dh.fetdim
        ssignals.emit(self, "ProjectionToChange", coord, channel,
                 self.projection[coord][1])
        
    def _select_feature_getter(self, coord, fet):
        """Return the callback function for the feature selection."""
        return lambda *args: self.select_feature(coord, fet)
        
    def _select_channel_getter(self, coord):
        """Return the callback function for the channel selection."""
        return lambda channel: self.select_channel(channel, coord)
        
    def create_feature_widget(self, coord=0):
        # coord => (channel, feature)
        self.projection = [(0, 0), (0, 1)]
        
        gridLayout = QtGui.QGridLayout()
        gridLayout.setSpacing(0)
        # HACK: pyside does not have this function
        if hasattr(gridLayout, 'setMargin'):
            gridLayout.setMargin(0)
        
        # channel selection
        comboBox = QtGui.QComboBox(self)
        comboBox.setEditable(True)
        comboBox.setInsertPolicy(QtGui.QComboBox.NoInsert)
        comboBox.addItems(["%d" % i for i in xrange(self.dh.nchannels)])
        comboBox.addItems(["Extra %d" % i for i in xrange(self.dh.nextrafet)])
        comboBox.editTextChanged.connect(self._select_channel_getter(coord), QtCore.Qt.UniqueConnection)
        # comboBox.currentIndexChanged.connect(self._select_channel_getter(coord), QtCore.Qt.UniqueConnection)
        # comboBox.setFocusPolicy(QtCore.Qt.ClickFocus)
        self.channel_box[coord] = comboBox
        gridLayout.addWidget(comboBox, 0, 0, 1, self.dh.fetdim)
        
        # create 3 buttons for selecting the feature
        widths = [30] * self.dh.fetdim
        labels = ['PC%d' % i for i in xrange(1, self.dh.fetdim + 1)]
        
        # ensure exclusivity of the group of buttons
        pushButtonGroup = QtGui.QButtonGroup(self)
        for i in xrange(len(labels)):
            # selecting feature i
            pushButton = QtGui.QPushButton(labels[i], self)
            pushButton.setCheckable(True)
            if coord == i:
                pushButton.setChecked(True)
            pushButton.setMaximumSize(QtCore.QSize(widths[i], 20))
            pushButton.clicked.connect(self._select_feature_getter(coord, i), QtCore.Qt.UniqueConnection)
            pushButtonGroup.addButton(pushButton, i)
            self.feature_buttons[coord][i] = pushButton
            gridLayout.addWidget(pushButton, 1, i)
        
        return gridLayout
        
    def create_nspikes_viewer(self):
        self.nspikes_viewer = QtGui.QLabel("", self)
        return self.nspikes_viewer
        
    def get_nspikes_text(self, nspikes, nspikes_highlighted):
        return "Spikes: %d. Highlighted: %d." % (nspikes, nspikes_highlighted)
        
    def update_nspikes_viewer(self, nspikes, nspikes_highlighted):
        text = self.get_nspikes_text(nspikes, nspikes_highlighted)
        self.nspikes_viewer.setText(text)
        
    def update_feature_widget(self):
        for coord in [0, 1]:
            comboBox = self.channel_box[coord]
            # update the channels/features list only if necessary
            if comboBox.count() != self.dh.nchannels + self.dh.nextrafet:
                comboBox.blockSignals(True)
                comboBox.clear()
                comboBox.addItems(["%d" % i for i in xrange(self.dh.nchannels)])
                comboBox.addItems(["Extra %d" % i for i in xrange(self.dh.nextrafet)])
                comboBox.blockSignals(False)
        
    def create_controller(self):
        box = super(FeatureWidget, self).create_controller()
        
        # coord => channel combo box
        self.channel_box = [None, None]
        # coord => (butA, butB, butC)
        self.feature_buttons = [[None] * self.dh.fetdim, [None] * self.dh.fetdim]
        
        # add navigation toolbar
        self.toolbar = self.create_toolbar()
        box.addWidget(self.toolbar)

        # add number of spikes
        self.nspikes_viewer = self.create_nspikes_viewer()
        box.addWidget(self.nspikes_viewer)
        
        # add feature widget
        self.feature_widget1 = self.create_feature_widget(0)
        box.addLayout(self.feature_widget1)
        
        # add feature widget
        self.feature_widget2 = self.create_feature_widget(1)
        box.addLayout(self.feature_widget2)
        
        self.setTabOrder(self.channel_box[0], self.channel_box[1])
        
        return box
    
    def set_navigation(self):
        self.view.set_interaction_mode(FeatureNavigationBindings)
    
    def set_selection(self):
        self.view.set_interaction_mode(FeatureSelectionBindings)
    
    
        


