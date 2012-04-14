package cbm

class Indicaciones {
	String nombre
	static constraints = {
	   nombre unique:true, blank:false
	}
	String toString(){nombre}
}
