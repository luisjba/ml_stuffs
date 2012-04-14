package cbm

class Presentacion {
	String nombre
    static constraints = {
    	nombre unique:true, blank:false
	}
	String toString(){nombre}
}
