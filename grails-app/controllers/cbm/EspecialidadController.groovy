package cbm

import org.springframework.dao.DataIntegrityViolationException

class EspecialidadController {
	def scaffold = true
	/*
    static allowedMethods = [save: "POST", update: "POST", delete: "POST"]

    def index() {
        redirect(action: "list", params: params)
    }

    def list() {
        params.max = Math.min(params.max ? params.int('max') : 10, 100)
        [especialidadInstanceList: Especialidad.list(params), especialidadInstanceTotal: Especialidad.count()]
    }

    def create() {
        [especialidadInstance: new Especialidad(params)]
    }

    def save() {
        def especialidadInstance = new Especialidad(params)
        if (!especialidadInstance.save(flush: true)) {
            render(view: "create", model: [especialidadInstance: especialidadInstance])
            return
        }

		flash.message = message(code: 'default.created.message', args: [message(code: 'especialidad.label', default: 'Especialidad'), especialidadInstance.id])
        redirect(action: "show", id: especialidadInstance.id)
    }

    def show() {
        def especialidadInstance = Especialidad.get(params.id)
        if (!especialidadInstance) {
			flash.message = message(code: 'default.not.found.message', args: [message(code: 'especialidad.label', default: 'Especialidad'), params.id])
            redirect(action: "list")
            return
        }

        [especialidadInstance: especialidadInstance]
    }

    def edit() {
        def especialidadInstance = Especialidad.get(params.id)
        if (!especialidadInstance) {
            flash.message = message(code: 'default.not.found.message', args: [message(code: 'especialidad.label', default: 'Especialidad'), params.id])
            redirect(action: "list")
            return
        }

        [especialidadInstance: especialidadInstance]
    }

    def update() {
        def especialidadInstance = Especialidad.get(params.id)
        if (!especialidadInstance) {
            flash.message = message(code: 'default.not.found.message', args: [message(code: 'especialidad.label', default: 'Especialidad'), params.id])
            redirect(action: "list")
            return
        }

        if (params.version) {
            def version = params.version.toLong()
            if (especialidadInstance.version > version) {
                especialidadInstance.errors.rejectValue("version", "default.optimistic.locking.failure",
                          [message(code: 'especialidad.label', default: 'Especialidad')] as Object[],
                          "Another user has updated this Especialidad while you were editing")
                render(view: "edit", model: [especialidadInstance: especialidadInstance])
                return
            }
        }

        especialidadInstance.properties = params

        if (!especialidadInstance.save(flush: true)) {
            render(view: "edit", model: [especialidadInstance: especialidadInstance])
            return
        }

		flash.message = message(code: 'default.updated.message', args: [message(code: 'especialidad.label', default: 'Especialidad'), especialidadInstance.id])
        redirect(action: "show", id: especialidadInstance.id)
    }

    def delete() {
        def especialidadInstance = Especialidad.get(params.id)
        if (!especialidadInstance) {
			flash.message = message(code: 'default.not.found.message', args: [message(code: 'especialidad.label', default: 'Especialidad'), params.id])
            redirect(action: "list")
            return
        }

        try {
            especialidadInstance.delete(flush: true)
			flash.message = message(code: 'default.deleted.message', args: [message(code: 'especialidad.label', default: 'Especialidad'), params.id])
            redirect(action: "list")
        }
        catch (DataIntegrityViolationException e) {
			flash.message = message(code: 'default.not.deleted.message', args: [message(code: 'especialidad.label', default: 'Especialidad'), params.id])
            redirect(action: "show", id: params.id)
        }
    }
*/
}
