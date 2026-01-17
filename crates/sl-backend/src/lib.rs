use sl_ir::Module;

pub trait Backend {
    fn name(&self) -> &str;
    fn compile(&self, module: &Module, out_path: &str) -> Result<(), String>;
    fn run(&self, artifact_path: &str) -> Result<(), String>;
}
