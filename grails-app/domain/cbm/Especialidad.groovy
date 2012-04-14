package cbm

class Especialidad {
	String nombre
	static hasMany = [medicamentos:Medicamento]
    static constraints = {
		nombre unique:true, blank:false
    }
	String toString(){nombre}
}
