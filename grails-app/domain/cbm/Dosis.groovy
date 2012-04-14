package cbm

class Dosis {
	String nombre
	static constraints = {
	    nombre unique:true, blank:false
	}
	String toString(){nombre}
}