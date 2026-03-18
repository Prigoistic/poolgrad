use crate::autograd::node::Node;

#[allow(dead_code)]
pub struct Graph {
    pub nodes: Vec<Node>,

}

#[allow(dead_code)]
impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
        }
    }

    pub fn add_node(&mut self, node:Node) {
        self.nodes.push(node);
    }

}