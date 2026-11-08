pub mod analysis;                                                                                                                                                                                               
pub mod atlas;                                                                                                                                                                                                  
pub mod markers;                                                                                                                                                                                                
pub mod renderer;                                                                                                                                                                                               
pub mod tts;                                                                                                                                                                                                    
pub mod visemes;                                                                                                                                                                                                
                                                                                                                                                                                                                
/// Placeholder module so the crate compiles before the first real component lands.                                                                                                                             
pub mod placeholder {                                                                                                                                                                                           
    /// Returns the crate version.                                                                                                                                                                              
    pub fn version() -> &'static str {                                                                                                                                                                          
        env!("CARGO_PKG_VERSION")                                                                                                                                                                               
    }                                                                                                                                                                                                           
}                                                                                                                                                                                                               
                                                                                                                                                                                                                
#[cfg(test)]                                                                                                                                                                                                    
mod tests {                                                                                                                                                                                                     
    #[test]                                                                                                                                                                                                     
    fn version_is_set() {                                                                                                                                                                                       
        assert!(!super::placeholder::version().is_empty());                                                                                                                                                     
    }                                                                                                                                                                                                           
}
